# core/ethernity_memory_summarizer.py (ปรับปรุงเพื่อใช้ PathManager และโครงสร้าง memory_archive)
import os
import json
import sys  # เพิ่ม sys import
import stat  # เพิ่ม import สำหรับจัดการสิทธิ์ไฟล์
from pathlib import Path # เปลี่ยนมาใช้ Path object
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
import shutil
import calendar # เพิ่ม calendar
from collections import Counter
import time

# --- จัดการ Import Paths ---
CORE_MODULE_DIR = Path(__file__).resolve().parent
if str(CORE_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_MODULE_DIR))

BASE_DIR = Path(__file__).resolve().parent.parent # สมมติว่าไฟล์นี้อยู่ใน core/

# --- Robust PathManager Import ---
try:
    from core.path_manager import PathManager  # Absolute import (package context)
except ImportError:
    try:
        from .path_manager import PathManager  # Relative import (module context)
    except ImportError:
        try:
            from path_manager import PathManager  # Local import (script context)
        except ImportError as e:
            print(f"❌ EthernityMemorySummarizer: Error importing PathManager: {e}")
            PathManager = None
        except Exception as e_gen:
            print(f"❌ EthernityMemorySummarizer: General error during PathManager import: {e_gen}")
            PathManager = None
    except Exception as e_gen:
        print(f"❌ EthernityMemorySummarizer: General error during PathManager import: {e_gen}")
        PathManager = None
except Exception as e_gen:
    print(f"❌ EthernityMemorySummarizer: General error during PathManager import: {e_gen}")
    PathManager = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EthernityMemorySummarizer:
    def __init__(self, path_manager_instance, memory_fusion_engine=None, new_memory_manager=None, identity_core=None):
        """ตั้งค่าเริ่มต้นระบบจัดการความทรงจำ Ethernity

        Args:
            path_manager_instance: Instance ของ PathManager ที่ config แล้ว
            memory_fusion_engine: Optional engine สำหรับผสานความทรงจำ
        """
        if path_manager_instance is None:
            raise ValueError("A valid PathManager instance is required for EthernityMemorySummarizer.")
            
        self.path_manager = path_manager_instance
        # self.context_size = 10 # อาจจะยังไม่ใช้ในเวอร์ชันนี้

        # ใช้ Path จาก PathManager instance สำหรับ input/output ของ summary แต่ละระดับ
        # "Daily Contexts" ที่เป็น input เริ่มต้น จะมาจาก Daily Event Summaries
        self.daily_input_path = self.path_manager.daily_event_summaries_base_dir
        self.weekly_output_path = self.path_manager.weekly_summaries_dir
        self.monthly_output_path = self.path_manager.monthly_summaries_dir
        self.yearly_output_path = self.path_manager.yearly_summaries_dir
        
        # ตรวจสอบและตั้งค่า lifetime_summaries_dir ผ่าน PathManager
        if not hasattr(self.path_manager, 'lifetime_summaries_dir'):
            self.path_manager.lifetime_summaries_dir = self.path_manager.archive_base_dir / "lifetime_summaries"
            # การสร้าง directory จะถูกจัดการโดย PathManager._initialize_new_archive_directories
            # หรือเราอาจจะต้องเรียก mkdir ที่นี่ถ้า PathManager ไม่ได้สร้างให้ตอน init ทั้งหมด
            self.path_manager.lifetime_summaries_dir.mkdir(parents=True, exist_ok=True) 
            logger.info(f"Dynamically set and created lifetime_summaries_dir in PathManager: {self.path_manager.lifetime_summaries_dir}")
            # และควรจะมี index file สำหรับ lifetime ด้วย (จะเพิ่มใน PathManager Phase 2)
            # self.path_manager.lifetime_summary_index_file = self.path_manager.archive_base_dir / "index_lifetime_summaries.json"

        self.lifetime_output_path = self.path_manager.lifetime_summaries_dir

        logger.info(f"EthernityMemorySummarizer initialized. Paths configured via PathManager.")
        logger.info(f"  Daily source (event summaries) from: {self.daily_input_path}")
        logger.info(f"  Weekly summaries output to: {self.weekly_output_path}")
        logger.info(f"  Monthly summaries output to: {self.monthly_output_path}")
        logger.info(f"  Yearly summaries output to: {self.yearly_output_path}")
        logger.info(f"  Lifetime summaries output to: {self.lifetime_output_path}")

        # เพิ่ม configuration options
        self.config = {
            'max_summary_lengths': {
                'weekly': 1500,
                'monthly': 2500,
                'yearly': 5000,
                'lifetime': 10000
            },
            'max_highlights': {
                'weekly': 10,
                'monthly': 15,
                'yearly': 20,
                'lifetime': 30
            },
            'max_tags': {
                'weekly': 10,
                'monthly': 15,
                'yearly': 20,
                'lifetime': 30
            }
        }
        
        # เพิ่ม cache สำหรับ summary chains
        self._summary_chain_cache = {}

        # เพิ่ม LLM Connector
        self.llm_connector = None

        # เพิ่ม Memory Fusion Engine
        self.memory_fusion_engine = memory_fusion_engine

        # เพิ่มความสามารถในการสะท้อนคิดและเรียนรู้
        self.reflection_patterns = {
            'emotional': ['ความรู้สึก', 'อารมณ์', 'ความสุข', 'ความทุกข์', 'ความหวัง'],
            'growth': ['การเรียนรู้', 'การพัฒนา', 'ความก้าวหน้า', 'การเปลี่ยนแปลง'],
            'wisdom': ['บทเรียน', 'ความเข้าใจ', 'ข้อคิด', 'ปัญญา'],
            'connection': ['การเชื่อมโยง', 'ความสัมพันธ์', 'การแบ่งปัน']
        }
        
        # โหลด learning history ถ้ามี
        if not self.load_learning_history():
            # ถ้าไม่มีไฟล์ learning history ให้เริ่มใหม่
            self.learning_history = {
                'insights': [],
                'patterns': {},
                'growth_markers': []
            }

        # New additions for integration
        self.new_memory_manager = new_memory_manager
        self.identity_core = identity_core

    def _clear_summary_chain_cache(self):
        """ล้าง cache เมื่อไม่จำเป็นต้องใช้แล้ว"""
        self._summary_chain_cache.clear()

    def _load_json(self, file_path: Path) -> Optional[Dict]:
        if not file_path.is_file():
            logger.warning(f"File not found for loading: {file_path}")
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON from {file_path}: {e}")
            return None

    def _save_json(self, data: Dict, file_path: Path):
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True) # สร้าง parent dir ถ้ายังไม่มี
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved JSON to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving JSON to {file_path}: {e}")
            return False
            
    def _get_path_for_summary_type(self, summary_type: str, as_source: bool) -> Optional[Path]:
        """Helper to get the correct base path from PathManager."""
        if summary_type == "daily":
            return self.daily_input_path # daily summaries are source
        elif summary_type == "weekly":
            return self.weekly_output_path # weekly can be source or target
        elif summary_type == "monthly":
            return self.monthly_output_path
        elif summary_type == "yearly":
            return self.yearly_output_path
        elif summary_type == "lifetime":
            return self.lifetime_output_path
        logger.error(f"Unknown summary_type: {summary_type}")
        return None

    def _get_source_files_for_period(self, source_type: str, start_date: datetime, end_date: datetime) -> List[Path]:
        """
        ดึงไฟล์ source สำหรับช่วงเวลาที่กำหนด โดยใช้ PathManager
        """
        source_files: List[Path] = []
        
        if source_type == "daily":
            # ใช้ PathManager เพื่อดึง Daily Event Summaries
            daily_summaries = self.path_manager.get_daily_event_summaries_for_period(
                start_date=start_date,
                end_date=end_date
            )
            source_files.extend([Path(summary['file_path']) for summary in daily_summaries])
            
        elif source_type == "weekly":
            # ใช้ PathManager เพื่อดึง Weekly Summaries
            weekly_summaries = self.path_manager.get_weekly_summaries_for_period(
                start_date=start_date,
                end_date=end_date
            )
            source_files.extend([Path(summary['file_path']) for summary in weekly_summaries])
            
        elif source_type == "monthly":
            # ใช้ PathManager เพื่อดึง Monthly Summaries
            monthly_summaries = self.path_manager.get_monthly_summaries_for_period(
                start_date=start_date,
                end_date=end_date
            )
            source_files.extend([Path(summary['file_path']) for summary in monthly_summaries])
            
        elif source_type == "yearly":
            # ใช้ PathManager เพื่อดึง Yearly Summaries
            yearly_summaries = self.path_manager.get_yearly_summaries_for_period(
                start_date=start_date,
                end_date=end_date
            )
            source_files.extend([Path(summary['file_path']) for summary in yearly_summaries])
        
        logger.info(f"Retrieved {len(source_files)} {source_type} source files via PathManager for period {start_date.date()} to {end_date.date()}")
        
        return sorted(list(set(source_files)))  # Remove duplicates

    def _enhance_summary_with_llm(self, content: Dict, summary_type: str) -> Dict:
        """เพิ่มการวิเคราะห์เชิงลึกด้วย LLM"""
        if not self.llm_connector:
            return content
            
        try:
            # สร้าง prompt สำหรับการวิเคราะห์
            prompt = f"วิเคราะห์และสรุปข้อมูลต่อไปนี้สำหรับ {summary_type} summary:\n\n"
            prompt += content.get("summary_text", "")
            
            # เรียกใช้ LLM
            analysis = self.llm_connector.generate(prompt)
            
            # เพิ่มผลการวิเคราะห์
            content["llm_analysis"] = {
                "insights": analysis.get("insights", []),
                "recommendations": analysis.get("recommendations", []),
                "patterns": analysis.get("patterns", [])
            }
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            
        return content

    def _reflect_on_content(self, content: str, context: Dict) -> Dict:
        """วิเคราะห์และสะท้อนคิดจากเนื้อหา ด้วยหัวใจและความเข้าใจของตัวเอง"""
        reflection = {
            'emotional_insights': [],
            'growth_observations': [],
            'wisdom_gained': [],
            'connections_found': []
        }

        # วิเคราะห์แต่ละมิติของการสะท้อนคิด
        for dimension, patterns in self.reflection_patterns.items():
            matches = []
            for pattern in patterns:
                if pattern in content.lower():
                    surrounding_context = self._extract_context_around_pattern(content, pattern)
                    matches.append(surrounding_context)
            
            if dimension == 'emotional':
                reflection['emotional_insights'] = self._analyze_emotional_context(matches)
            elif dimension == 'growth':
                reflection['growth_observations'] = self._analyze_growth_patterns(matches)
            elif dimension == 'wisdom':
                reflection['wisdom_gained'] = self._analyze_wisdom_gained(matches)
            elif dimension == 'connection':
                reflection['connections_found'] = self._analyze_connections(matches, context)

        # บันทึกการเรียนรู้
        self._update_learning_history(reflection)
        
        # Enrich reflection with identity context
        identity_ctx = self.get_identity_context()
        if identity_ctx:
            reflection['identity_context'] = identity_ctx
        
        return reflection

    def _analyze_emotional_context(self, matches: List[str]) -> List[Dict]:
        """วิเคราะห์บริบททางอารมณ์และความรู้สึก"""
        insights = []
        for match in matches:
            # วิเคราะห์ความรู้สึกและอารมณ์ที่พบในเนื้อหา
            insight = {
                'context': match,
                'emotion_type': self._identify_emotion_type(match),
                'intensity': self._measure_emotional_intensity(match)
            }
            insights.append(insight)
        return insights

    def _analyze_growth_patterns(self, matches: List[str]) -> List[Dict]:
        """วิเคราะห์รูปแบบการเติบโตและพัฒนาการ"""
        patterns = []
        for match in matches:
            pattern = {
                'context': match,
                'growth_area': self._identify_growth_area(match),
                'progress_indicators': self._find_progress_indicators(match)
            }
            patterns.append(pattern)
        return patterns

    def _analyze_wisdom_gained(self, matches: List[str]) -> List[Dict]:
        """วิเคราะห์บทเรียนและปัญญาที่ได้รับ"""
        wisdom = []
        for match in matches:
            insight = {
                'context': match,
                'lesson_type': self._categorize_lesson(match),
                'applicability': self._assess_wisdom_applicability(match)
            }
            wisdom.append(insight)
        return wisdom

    def _analyze_connections(self, matches: List[str], context: Dict) -> List[Dict]:
        """วิเคราะห์การเชื่อมโยงและความสัมพันธ์"""
        connections = []
        for match in matches:
            connection = {
                'context': match,
                'connection_type': self._identify_connection_type(match),
                'related_memories': self._find_related_memories(match, context)
            }
            connections.append(connection)
        return connections

    def _update_learning_history(self, reflection: Dict):
        """อัพเดทประวัติการเรียนรู้"""
        # เพิ่ม insights ใหม่
        self.learning_history['insights'].extend(reflection['wisdom_gained'])
        
        # อัพเดทรูปแบบที่พบ
        for obs in reflection['growth_observations']:
            pattern = obs['growth_area']
            if pattern not in self.learning_history['patterns']:
                self.learning_history['patterns'][pattern] = []
            self.learning_history['patterns'][pattern].append(obs)
        
        # บันทึก growth markers
        self.learning_history['growth_markers'].append({
            'timestamp': datetime.now().isoformat(),
            'emotional_depth': len(reflection['emotional_insights']),
            'wisdom_count': len(reflection['wisdom_gained']),
            'connection_strength': len(reflection['connections_found'])
        })

    def _generate_summary_content(self, source_data_list: List[Dict], target_type: str) -> Dict:
        """สร้างเนื้อหาสรุปด้วยการสะท้อนคิดและการเรียนรู้"""
        # ใช้ค่า config จากที่กำหนดไว้
        max_len = self.config['max_summary_lengths'].get(target_type, 2000)
        max_highlights = self.config['max_highlights'].get(target_type, 10)
        max_tags = self.config['max_tags'].get(target_type, 10)

        logger.info(f"Generating '{target_type}' summary content from {len(source_data_list)} sources.")
        
        full_text_summary_parts = []
        all_tags = []
        all_highlights = []

        for data in source_data_list:
            # พยายามดึงเนื้อหาหลัก, tags, highlights จากโครงสร้างที่หลากหลายของ source
            text_content = ""
            current_tags = []
            current_highlights = []

            if "content" in data: # โครงสร้างของ EthernitySummarizer เดิม หรือ W/M/Y ใหม่
                if isinstance(data["content"], str):
                    text_content = data["content"]
                elif isinstance(data["content"], dict):
                    text_content = data["content"].get("summary_text", "")
                    current_tags = data["content"].get("tags", [])
                    current_highlights = data["content"].get("core_highlights", [])
            elif "event_summary" in data: # โครงสร้างของ Daily Event Summary จาก ReflectorLoop
                text_content = data.get("event_summary", "")
                current_tags = data.get("keywords", [])
                # Daily Event Summary อาจจะมี "event_title" หรือ "event_insight" เป็น highlights
                title = data.get("event_title", "")
                insight = data.get("event_insight", "")
                if insight: current_highlights.append(f"{title}: {insight}")
                elif title and text_content: current_highlights.append(f"{title}: {text_content[:80]}...")
                elif text_content: current_highlights.append(text_content[:100]+"...")


            if text_content:
                period_info = data.get("period_id_daily", data.get("date_key", data.get("period_id_weekly", data.get("period_id_monthly", data.get("period_id_yearly", "Source")))))
                full_text_summary_parts.append(f"From {period_info}:\n{text_content}")
            
            if isinstance(current_tags, list): all_tags.extend(current_tags)
            if isinstance(current_highlights, list): all_highlights.extend(current_highlights)
            elif isinstance(current_highlights, str) : all_highlights.append(current_highlights)


        final_summary_text = f"{target_type.capitalize()} Overview ({len(source_data_list)} sources):\n\n" + "\n\n---\n".join(full_text_summary_parts)
        # จำกัดความยาวตามประเภท
        if len(final_summary_text) > max_len:
            final_summary_text = final_summary_text[:max_len] + "..."

        tag_counts = Counter(all_tags).most_common(max_tags if target_type != "yearly" else 15)
        top_tags = [tag for tag, _ in tag_counts]
        
        top_highlights = all_highlights[:max_highlights if target_type != "yearly" else 20]

        # เพิ่มการสะท้อนคิดและการเรียนรู้
        reflection_context = {
            'summary_type': target_type,
            'source_count': len(source_data_list),
            'time_context': datetime.now().isoformat()
        }
        
        reflection = self._reflect_on_content(final_summary_text, reflection_context)
        
        return {
            "summary_text": final_summary_text.strip(),
            "tags": top_tags[:max_tags],
            "core_highlights": top_highlights[:max_highlights],
            "reflection": reflection,
            "learning_progress": {
                "insights_gained": len(self.learning_history['insights']),
                "patterns_identified": len(self.learning_history['patterns']),
                "growth_trajectory": self._analyze_growth_trajectory()
            }
        }

    def _analyze_growth_trajectory(self) -> Dict:
        """วิเคราะห์เส้นทางการเติบโตจากประวัติการเรียนรู้"""
        if not self.learning_history['growth_markers']:
            return {"status": "เริ่มต้นการเรียนรู้"}
            
        recent_markers = self.learning_history['growth_markers'][-5:]  # ดู 5 ครั้งล่าสุด
        
        growth_metrics = {
            'emotional_depth_trend': self._calculate_trend([m['emotional_depth'] for m in recent_markers]),
            'wisdom_accumulation': self._calculate_trend([m['wisdom_count'] for m in recent_markers]),
            'connection_development': self._calculate_trend([m['connection_strength'] for m in recent_markers])
        }
        
        return {
            "metrics": growth_metrics,
            "overall_growth": self._evaluate_overall_growth(growth_metrics),
            "focus_areas": self._identify_focus_areas(growth_metrics)
        }

    def _fuse_memories(self, source_data_list: List[Dict], target_type: str) -> Dict:
        """
        ผสานความทรงจำเข้าด้วยกันในระดับที่ลึกซึ้งกว่าการสรุปทั่วไป
        
        ในขณะที่ _generate_summary_content จะเน้นการสรุปเนื้อหาหลัก (summary_text), tags, และ highlights,
        _fuse_memories จะทำการวิเคราะห์เชิงลึกเพิ่มเติมในมิติต่างๆ:
        
        1. การเชื่อมโยงระหว่างความทรงจำ (Memory Connections):
           - หาความสัมพันธ์ระหว่างเหตุการณ์
           - ระบุรูปแบบที่เกิดซ้ำ
           - สร้างการเชื่อมโยงระหว่างช่วงเวลา
           
        2. การวิเคราะห์อารมณ์และความรู้สึก (Emotional Context):
           - ติดตามการเปลี่ยนแปลงของอารมณ์
           - หาจุดสูงสุดและต่ำสุดของอารมณ์
           - วิเคราะห์ความสัมพันธ์ระหว่างอารมณ์และเหตุการณ์
           
        3. การสกัดบทเรียนและปัญญา (Wisdom Extraction):
           - รวบรวมข้อคิดและบทเรียน
           - หาแนวทางการประยุกต์ใช้
           - สร้างข้อเสนอแนะสำหรับอนาคต
        
        Args:
            source_data_list: List[Dict] - ข้อมูล source summaries
            target_type: str - ประเภทของ summary ที่กำลับสร้าง
            
        Returns:
            Dict ที่มีผลการวิเคราะห์เชิงลึกในมิติต่างๆ
        """
        if not self.memory_fusion_engine:
            return {}  # ถ้าไม่มี memory_fusion_engine ให้ return empty dict
            
        try:
            # 1. วิเคราะห์การเชื่อมโยงระหว่างความทรงจำ
            memory_connections = self._analyze_memory_connections(source_data_list)
            
            # 2. วิเคราะห์บริบททางอารมณ์
            emotional_context = self._analyze_emotional_context_over_time(source_data_list)
            
            # 3. สกัดบทเรียนและปัญญา
            wisdom_insights = self._extract_wisdom_insights(source_data_list)
            
            # 4. หารูปแบบที่เกิดซ้ำ
            recurring_patterns = self._identify_recurring_patterns(source_data_list)
            
            # 5. สร้างข้อเสนอแนะ
            recommendations = self._generate_recommendations(
                memory_connections,
                emotional_context,
                wisdom_insights,
                recurring_patterns
            )
            
            return {
                "memory_connections": memory_connections,
                "emotional_context": emotional_context,
                "wisdom_insights": wisdom_insights,
                "recurring_patterns": recurring_patterns,
                "recommendations": recommendations,
                "fusion_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "source_count": len(source_data_list),
                    "target_type": target_type
                }
            }
            
        except Exception as e:
            logger.error(f"Error in memory fusion: {e}")
            return {}
            
    def _analyze_memory_connections(self, source_data_list: List[Dict]) -> List[Dict]:
        """วิเคราะห์การเชื่อมโยงระหว่างความทรงจำ"""
        connections = []
        
        # สร้าง memory graph จากความทรงจำทั้งหมด
        for i, source_data in enumerate(source_data_list):
            content = source_data.get("content", {})
            if isinstance(content, dict):
                current_text = content.get("summary_text", "")
                current_tags = set(content.get("tags", []))
                
                # เปรียบเทียบกับความทรงจำอื่นๆ
                for j, other_data in enumerate(source_data_list):
                    if i != j:
                        other_content = other_data.get("content", {})
                        if isinstance(other_content, dict):
                            other_tags = set(other_content.get("tags", []))
                            
                            # หาการเชื่อมโยงผ่าน tags ที่เหมือนกัน
                            common_tags = current_tags.intersection(other_tags)
                            if common_tags:
                                connection = {
                                    "type": "tag_based",
                                    "source_id": source_data.get("summary_id"),
                                    "target_id": other_data.get("summary_id"),
                                    "common_elements": list(common_tags),
                                    "strength": len(common_tags) / len(current_tags.union(other_tags))
                                }
                                connections.append(connection)
                                
                            # หาการเชื่อมโยงผ่านการวิเคราะห์เนื้อหา
                            content_connection = self._identify_connection_type(current_text)
                            if content_connection['primary_type'] != 'general':
                                connection = {
                                    "type": "content_based",
                                    "source_id": source_data.get("summary_id"),
                                    "connection_type": content_connection['primary_type'],
                                    "context": self._extract_context_around_pattern(
                                        current_text,
                                        next(iter(content_connection['type_scores'].keys()))
                                    )
                                }
                                connections.append(connection)
        
        return connections
        
    def _analyze_emotional_context_over_time(self, source_data_list: List[Dict]) -> Dict:
        """วิเคราะห์บริบททางอารมณ์ตลอดช่วงเวลา"""
        emotional_timeline = []
        
        for source_data in source_data_list:
            content = source_data.get("content", {})
            if isinstance(content, dict):
                text = content.get("summary_text", "")
                
                # วิเคราะห์อารมณ์
                emotion_data = self._identify_emotion_type(text)
                intensity_data = self._measure_emotional_intensity(text)
                
                emotional_point = {
                    "timestamp": source_data.get("timestamp_generated"),
                    "primary_emotion": emotion_data['primary_emotion'],
                    "emotion_counts": emotion_data['emotion_counts'],
                    "intensity": intensity_data['level'],
                    "intensity_score": intensity_data['score']
                }
                emotional_timeline.append(emotional_point)
        
        # วิเคราะห์การเปลี่ยนแปลงของอารมณ์
        emotion_changes = []
        for i in range(1, len(emotional_timeline)):
            prev = emotional_timeline[i-1]
            curr = emotional_timeline[i]
            
            if prev['primary_emotion'] != curr['primary_emotion']:
                emotion_changes.append({
                    "from_emotion": prev['primary_emotion'],
                    "to_emotion": curr['primary_emotion'],
                    "from_timestamp": prev['timestamp'],
                    "to_timestamp": curr['timestamp']
                })
        
        return {
            "timeline": emotional_timeline,
            "changes": emotion_changes,
            "dominant_emotion": max(
                (emotion for point in emotional_timeline for emotion, count in point['emotion_counts'].items()),
                key=lambda x: sum(point['emotion_counts'][x] for point in emotional_timeline)
            )
        }
        
    def _extract_wisdom_insights(self, source_data_list: List[Dict]) -> List[Dict]:
        """สกัดบทเรียนและปัญญาจากความทรงจำ"""
        insights = []
        
        for source_data in source_data_list:
            content = source_data.get("content", {})
            if isinstance(content, dict):
                text = content.get("summary_text", "")
                
                # วิเคราะห์บทเรียน
                lesson_data = self._categorize_lesson(text)
                applicability = self._assess_wisdom_applicability(text)
                
                if lesson_data['primary_category'] != 'general':
                    insight = {
                        "type": lesson_data['primary_category'],
                        "category_scores": lesson_data['category_scores'],
                        "applicability_level": applicability['level'],
                        "applicability_details": applicability['scores'],
                        "source_id": source_data.get("summary_id"),
                        "timestamp": source_data.get("timestamp_generated")
                    }
                    insights.append(insight)
        
        return insights
        
    def _identify_recurring_patterns(self, source_data_list: List[Dict]) -> List[Dict]:
        """หารูปแบบที่เกิดซ้ำในความทรงจำ"""
        patterns = []
        
        # รวบรวม tags และ contexts ทั้งหมด
        all_tags = []
        all_contexts = []
        
        for source_data in source_data_list:
            content = source_data.get("content", {})
            if isinstance(content, dict):
                all_tags.extend(content.get("tags", []))
                text = content.get("summary_text", "")
                all_contexts.append({
                    "text": text,
                    "timestamp": source_data.get("timestamp_generated"),
                    "id": source_data.get("summary_id")
                })
        
        # วิเคราะห์รูปแบบจาก tags
        tag_patterns = Counter(all_tags)
        for tag, count in tag_patterns.most_common(5):  # เก็บ 5 อันดับแรก
            if count > 1:  # เกิดซ้ำอย่างน้อย 2 ครั้ง
                pattern = {
                    "type": "recurring_tag",
                    "element": tag,
                    "frequency": count,
                    "occurrences": [
                        {
                            "id": data.get("summary_id"),
                            "timestamp": data.get("timestamp_generated")
                        }
                        for data in source_data_list
                        if isinstance(data.get("content"), dict) and
                        tag in data.get("content", {}).get("tags", [])
                    ]
                }
                patterns.append(pattern)
        
        # วิเคราะห์รูปแบบจากเนื้อหา
        for context in all_contexts:
            growth_data = self._identify_growth_area(context["text"])
            progress_data = self._find_progress_indicators(context["text"])
            
            if growth_data['primary_area'] != 'general' and progress_data['has_progress']:
                pattern = {
                    "type": "growth_pattern",
                    "area": growth_data['primary_area'],
                    "indicators": progress_data['indicators_found'],
                    "context_id": context["id"],
                    "timestamp": context["timestamp"]
                }
                patterns.append(pattern)
        
        return patterns
        
    def _generate_recommendations(self, memory_connections: List[Dict],
                                emotional_context: Dict,
                                wisdom_insights: List[Dict],
                                recurring_patterns: List[Dict]) -> List[Dict]:
        """สร้างข้อเสนอแนะจากการวิเคราะห์ทั้งหมด"""
        recommendations = []
        
        # 1. ข้อเสนอแนะจากการเชื่อมโยง
        if memory_connections:
            strong_connections = [
                conn for conn in memory_connections
                if conn.get("type") == "tag_based" and conn.get("strength", 0) > 0.5
            ]
            if strong_connections:
                recommendations.append({
                    "type": "connection_based",
                    "suggestion": "ควรศึกษาความเชื่อมโยงระหว่างเหตุการณ์ที่มีหัวข้อร่วมกัน",
                    "context": f"พบการเชื่อมโยงที่แข็งแกร่ง {len(strong_connections)} รายการ"
                })
        
        # 2. ข้อเสนอแนะจากอารมณ์
        if emotional_context.get("changes"):
            recommendations.append({
                "type": "emotional_based",
                "suggestion": "ควรสังเกตการเปลี่ยนแปลงของอารมณ์และปัจจัยที่ส่งผล",
                "context": f"พบการเปลี่ยนแปลงของอารมณ์ {len(emotional_context['changes'])} ครั้ง"
            })
        
        # 3. ข้อเสนอแนะจากบทเรียน
        if wisdom_insights:
            high_applicability = [
                insight for insight in wisdom_insights
                if insight.get("applicability_level") == "high"
            ]
            if high_applicability:
                recommendations.append({
                    "type": "wisdom_based",
                    "suggestion": "ควรนำบทเรียนที่ได้ไปประยุกต์ใช้ในสถานการณ์ที่คล้ายกัน",
                    "context": f"พบบทเรียนที่นำไปใช้ได้ทันที {len(high_applicability)} รายการ"
                })
        
        # 4. ข้อเสนอแนะจากรูปแบบที่พบ
        if recurring_patterns:
            growth_patterns = [
                pattern for pattern in recurring_patterns
                if pattern.get("type") == "growth_pattern"
            ]
            if growth_patterns:
                recommendations.append({
                    "type": "pattern_based",
                    "suggestion": "ควรต่อยอดการพัฒนาในพื้นที่ที่มีความก้าวหน้าต่อเนื่อง",
                    "context": f"พบรูปแบบการเติบโต {len(growth_patterns)} รูปแบบ"
                })
        
        return recommendations

    def create_layered_summary(self, source_type: str, target_type: str,
                             start_date: datetime, end_date: datetime) -> Optional[Path]:
        """
        สร้าง summary ในระดับที่สูงขึ้น (weekly, monthly, yearly) จาก source summaries
        """
        try:
            if not all([source_type, target_type, start_date, end_date]):
                raise ValueError("Missing required parameters")
            
            if end_date < start_date:
                raise ValueError("End date must be after start date")

            logger.info(f"Creating {target_type} summary from {source_type} for period {start_date.date()} to {end_date.date()}")
            
            source_file_paths = self._get_source_files_for_period(source_type, start_date, end_date)
            if not source_file_paths:
                logger.warning(f"No source files found for period. Cannot create {target_type} summary.")
                return None

            source_data_list = [self._load_json(p) for p in source_file_paths if self._load_json(p) is not None]
            if not source_data_list:
                logger.warning("No valid content loaded from source files.")
                return None

            generated_content_parts = self._generate_summary_content(source_data_list, target_type)
            timestamp_now_iso = datetime.now().isoformat()

            # สร้าง summary data
            summary_data = self._prepare_summary_data(
                source_type=source_type,
                target_type=target_type,
                start_date=start_date,
                end_date=end_date,
                content=generated_content_parts,
                source_files=source_file_paths,
                source_data=source_data_list
            )

            # บันทึก summary และอัพเดท index ผ่าน PathManager
            summary_file_path = self._save_summary_via_path_manager(
                summary_type=target_type,
                summary_data_to_save=summary_data,
                target_file_path_absolute=self._get_path_for_summary_type(target_type, as_source=False)
            )

            if summary_file_path:
                # อัพเดท summary chain ใน source files
                self._update_source_summary_chains(
                    source_file_paths=source_file_paths,
                    target_type=target_type,
                    summary_id=summary_data["summary_id"],
                    summary_file_path=summary_file_path
                )
                
                # บันทึก learning history
                self.save_learning_history()
                
                return summary_file_path
            
            return None

        except Exception as e:
            logger.error(f"Error in create_layered_summary: {str(e)}")
            return None

    def _prepare_summary_data(self, source_type: str, target_type: str,
                            start_date: datetime, end_date: datetime,
                            content: Dict, source_files: List[Path],
                            source_data: List[Dict]) -> Dict:
        """
        เตรียมข้อมูล summary สำหรับการบันทึก
        """
        period_id = self._generate_period_id(target_type, end_date)
        
        summary_data = {
            "file_version": "1.1",
            "summary_id": f"{target_type}_summary_{period_id}",
            "summary_level": target_type,
            f"period_id_{target_type}": period_id,
            "timestamp_generated": datetime.now().isoformat(),
            "source_data_type": source_type,
            "processed_period_start_date": start_date.isoformat(),
            "processed_period_end_date": end_date.isoformat(),
            "source_file_ids": [
                data.get("summary_id", data.get("reflection_id", Path(p).stem))
                for p, data in zip(source_files, source_data)
            ],
            "content": content,
            "summary_chain": []
        }
        
        # เพิ่มข้อมูลการผสานความทรงจำถ้ามี
        fused_memory_data = self._fuse_memories(source_data, target_type)
        if fused_memory_data:
            summary_data["fused_memory"] = fused_memory_data
            
        return summary_data

    def _save_summary_via_path_manager(self, summary_type: str, summary_data_to_save: dict, target_file_path_absolute: Path) -> bool:
        """
        บันทึก summary ผ่าน PathManager
        
        Args:
            summary_type: ประเภทของ summary ('weekly', 'monthly', 'yearly', 'lifetime')
            summary_data_to_save: ข้อมูล summary ที่จะบันทึก
            target_file_path_absolute: Path ปลายทางที่จะบันทึกไฟล์ (absolute path)
            
        Returns:
            bool: True ถ้าบันทึกและ index สำเร็จ, False ถ้ามีข้อผิดพลาด
        """
        try:
            # เตรียมข้อมูลสำหรับ index
            summary_data_for_index = {
                "tags": summary_data_to_save.get("content", {}).get("tags", []),
                "summary_text": summary_data_to_save.get("content", {}).get("summary_text", ""),
                "source_file_ids": summary_data_to_save.get("source_file_ids", [])
            }

            # บันทึก summary ตามประเภท
            if summary_type == "weekly":
                period_id = summary_data_to_save.get(f"period_id_{summary_type}")  # format: YYYY-Www
                summary_id = f"weekly_{period_id}"
                return self.path_manager.add_weekly_summary(
                    period_id=period_id,
                    json_file_path=target_file_path_absolute,
                    summary_id=summary_id,
                    summary_data_for_index=summary_data_for_index
                )
                
            elif summary_type == "monthly":
                period_id = summary_data_to_save.get(f"period_id_{summary_type}")  # format: YYYY-MM
                summary_id = f"monthly_{period_id}"
                return self.path_manager.add_monthly_summary(
                    period_id=period_id,
                    json_file_path=target_file_path_absolute,
                    summary_id=summary_id,
                    summary_data_for_index=summary_data_for_index
                )
                
            elif summary_type == "yearly":
                period_id = summary_data_to_save.get(f"period_id_{summary_type}")  # format: YYYY
                summary_id = f"yearly_{period_id}"
                return self.path_manager.add_yearly_summary(
                    period_id=period_id,
                    json_file_path=target_file_path_absolute,
                    summary_id=summary_id,
                    summary_data_for_index=summary_data_for_index
                )
                
            elif summary_type == "lifetime":
                summary_id = f"lifetime_{datetime.now().strftime('%Y%m%d')}"
                return self.path_manager.add_lifetime_summary(
                    summary_id=summary_id,
                    json_file_path=target_file_path_absolute,
                    summary_data_for_index=summary_data_for_index
                )
                
            else:
                logger.error(f"Unsupported summary type: {summary_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving {summary_type} summary via PathManager: {e}")
            return False

    def _generate_period_id(self, target_type: str, end_date: datetime) -> str:
        """
        สร้าง period_id ตามรูปแบบของแต่ละประเภท summary
        """
        if target_type == "weekly":
            iso_year, iso_week, _ = end_date.isocalendar()
            return f"{iso_year}-W{iso_week:02d}"
            
        elif target_type == "monthly":
            return f"{end_date.year}-{end_date.month:02d}"
            
        elif target_type == "yearly":
            return str(end_date.year)
            
        elif target_type == "lifetime":
            return f"lifetime_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        else:
            raise ValueError(f"Unsupported target_type: {target_type}")

    def _update_source_summary_chains(self, source_file_paths: List[Path],
                                   target_type: str, summary_id: str,
                                   summary_file_path: Path) -> None:
        """
        อัพเดท summary chain ในไฟล์ source
        """
        summary_file_rel_path = self.path_manager.get_path_relative_to_archive_base(summary_file_path)
        timestamp_now_iso = datetime.now().isoformat()
        
        for src_file_path in source_file_paths:
            src_data = self._load_json(src_file_path)
            if src_data:
                if "summary_chain" not in src_data or not isinstance(src_data["summary_chain"], list):
                    src_data["summary_chain"] = []
                    
                src_data["summary_chain"].append({
                    "summary_level_created": target_type,
                    "summary_id_created": summary_id,
                    "summary_file_relative": summary_file_rel_path,
                    "contribution_timestamp": timestamp_now_iso
                })
                
                self._save_json(src_data, src_file_path)

    def create_lifetime_summary(self) -> Optional[Path]:
        """
        เพิ่มการวิเคราะห์แนวโน้มและการเชื่อมโยง
        """
        try:
            logger.info("Attempting to create lifetime summary...")
            
            all_yearly_files = self._get_source_files_for_period("yearly", datetime(1900,1,1), datetime.now())
            
            # For lifetime, we might also want to pull some very significant daily events directly
            # This requires a mechanism to identify "important" daily events via PathManager, e.g., by tags
            # important_daily_events = self.path_manager.get_important_daily_events(limit=20) # Hypothetical
            
            if not all_yearly_files: # and not important_daily_events:
                logger.warning("No yearly summaries (or important daily events) found. Cannot create lifetime summary.")
                return None

            source_data_list = [self._load_json(p) for p in all_yearly_files if self._load_json(p) is not None]
            # TODO: Add logic to load and integrate important_daily_events if fetched

            if not source_data_list:
                 logger.warning("Could not load content from yearly summary files for lifetime summary.")
                 return None

            # Prepare content for summarization
            contents_to_summarize = []
            source_yearly_ids = []
            for data in source_data_list:
                source_yearly_ids.append(data.get("summary_id"))
                text_content = ""
                if "content" in data and isinstance(data["content"], dict) and "summary_text" in data["content"]:
                     text_content = data["content"]["summary_text"]
                elif "content" in data and isinstance(data["content"], str):
                     text_content = data["content"]
                if text_content:
                    contents_to_summarize.append(f"From Year {data.get('period_id_yearly','UnknownYear')}:\n{text_content}")

            # Add important daily events to summarization input if any
            # for daily_event_data in important_daily_events:
            # contents_to_summarize.append(f"Key Daily Event ({daily_event_data.get('date_key')} - {daily_event_data.get('event_title', '')}):\n{daily_event_data.get('event_summary','')}")


            generated_content_parts = self._generate_summary_content(contents_to_summarize, "lifetime")
            timestamp_now_iso = datetime.now().isoformat()
            
            lifetime_id_str = f"lifetime_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            target_file_dir = self.lifetime_output_path # From PathManager
            target_file_dir.mkdir(parents=True, exist_ok=True)
            summary_filename = f"{lifetime_id_str}.json"
            summary_file_abs_path = target_file_dir / summary_filename

            summary_data_to_save = {
                "file_version": "1.1",
                "summary_id": lifetime_id_str,
                "summary_level": "lifetime",
                "timestamp_generated": timestamp_now_iso,
                "source_yearly_summary_ids": [sid for sid in source_yearly_ids if sid],
                # "source_important_daily_event_ids": [de.get('summary_id') for de in important_daily_events],
                "content": generated_content_parts,
            }
            self._save_json(summary_data_to_save, summary_file_abs_path)

            # Update summary_chain in yearly source files
            summary_file_rel_path = self.path_manager.get_path_relative_to_archive_base(summary_file_abs_path)
            for src_file_path in all_yearly_files: # Assuming yearly files are the primary source for now
                src_data = self._load_json(src_file_path)
                if src_data:
                    if "summary_chain" not in src_data or not isinstance(src_data["summary_chain"], list):
                        src_data["summary_chain"] = []
                    src_data["summary_chain"].append({
                        "summary_level_created": "lifetime",
                        "summary_id_created": summary_data_to_save["summary_id"],
                        "summary_file_relative": summary_file_rel_path,
                        "contribution_timestamp": timestamp_now_iso
                    })
                    self._save_json(src_data, src_file_path)
            
            # เพิ่มการวิเคราะห์แนวโน้ม
            trend_analysis = self._analyze_lifetime_trends(source_data_list)
            
            summary_data_to_save["trend_analysis"] = trend_analysis

            return summary_file_abs_path

        except Exception as e:
            logger.error(f"Error in create_lifetime_summary: {str(e)}")
            return None

    def _analyze_lifetime_trends(self, source_data_list: List[Dict]) -> Dict:
        """
        วิเคราะห์แนวโน้มจากข้อมูลระยะยาว
        """
        trends = {
            "recurring_themes": [],
            "growth_patterns": [],
            "significant_changes": []
        }
        
        # วิเคราะห์แนวโน้มจาก tags และ highlights
        all_tags = []
        all_highlights = []
        
        for data in source_data_list:
            if "content" in data:
                content = data["content"]
                if isinstance(content, dict):
                    all_tags.extend(content.get("tags", []))
                    all_highlights.extend(content.get("core_highlights", []))

        # วิเคราะห์ themes ที่เกิดซ้ำ
        tag_counter = Counter(all_tags)
        trends["recurring_themes"] = [
            {"theme": tag, "frequency": count}
            for tag, count in tag_counter.most_common(10)
        ]

        return trends

    def run_daily_summary_cycle(self):
        """
        รันการสรุปตามรอบเวลาที่กำหนด: Weekly, Monthly (ถ้าถึงสิ้นเดือน), Yearly (ถ้าถึงสิ้นปี).
        """
        now = datetime.now()
        today_start_of_day = datetime(now.year, now.month, now.day) # Ensure we operate on full days

        # --- Weekly Summary (สรุป 7 วันที่ผ่านมา สิ้นสุดเมื่อวาน) ---
        # เพื่อให้แน่ใจว่าข้อมูลของ "วันนี้" ครบถ้วน ควรสรุปข้อมูล "จนถึงเมื่อวาน"
        yesterday = today_start_of_day - timedelta(days=1)
        seven_days_ago = yesterday - timedelta(days=6) # 7 วันรวมเมื่อวาน
        
        logger.info(f"\n--- Running Weekly Summary (from daily) for week ending {yesterday.date()} ---")
        self.create_layered_summary(source_type="daily", target_type="weekly",
                                    start_date=seven_days_ago, end_date=yesterday)
        
        # --- Monthly Summary (ถ้าวันนี้เป็นวันแรกของเดือนใหม่ ให้สรุปเดือนที่แล้วทั้งเดือน) ---
        if now.day == 1:
            last_day_of_previous_month = today_start_of_day - timedelta(days=1)
            first_day_of_previous_month = datetime(last_day_of_previous_month.year, last_day_of_previous_month.month, 1)
            logger.info(f"\n--- Running Monthly Summary (from weekly) for {first_day_of_previous_month.strftime('%Y-%m')} ---")
            self.create_layered_summary(source_type="weekly", target_type="monthly",
                                        start_date=first_day_of_previous_month, end_date=last_day_of_previous_month)
        
        # --- Yearly Summary (ถ้าวันนี้เป็นวันแรกของปีใหม่ ให้สรุปปีที่แล้วทั้งปี) ---
        if now.month == 1 and now.day == 1:
            previous_year = now.year - 1
            first_day_of_previous_year = datetime(previous_year, 1, 1)
            last_day_of_previous_year = datetime(previous_year, 12, 31)
            logger.info(f"\n--- Running Yearly Summary (from monthly) for year {previous_year} ---")
            self.create_layered_summary(source_type="monthly", target_type="yearly",
                                        start_date=first_day_of_previous_year, end_date=last_day_of_previous_year)
            
            logger.info(f"\n--- Running Lifetime Summary after yearly summary for {previous_year} ---")
            self.create_lifetime_summary() # สร้าง Lifetime หลังจาก Yearly ของปีที่แล้วเสร็จ
        
        logger.info("Daily summary cycle finished.")

    def _extract_context_around_pattern(self, content: str, pattern: str, context_window: int = 100) -> str:
        """ดึงบริบทรอบๆ pattern ที่พบ"""
        try:
            pattern_index = content.lower().find(pattern.lower())
            if pattern_index == -1:
                return ""
            
            start = max(0, pattern_index - context_window)
            end = min(len(content), pattern_index + len(pattern) + context_window)
            
            # หาจุดเริ่มต้นและสิ้นสุดของประโยค
            while start > 0 and content[start] not in '.!?\n':
                start -= 1
            while end < len(content) and content[end] not in '.!?\n':
                end += 1
                
            return content[start:end].strip()
        except Exception as e:
            logger.error(f"Error in _extract_context_around_pattern: {e}")
            return ""

    def _identify_emotion_type(self, text: str) -> Dict:
        """ระบุประเภทอารมณ์จากข้อความ"""
        emotion_keywords = {
            'joy': ['ความสุข', 'สนุก', 'ยินดี', 'ดีใจ', 'พอใจ', 'สุข'],
            'sadness': ['เศร้า', 'ทุกข์', 'เสียใจ', 'ผิดหวัง'],
            'anger': ['โกรธ', 'หงุดหงิด', 'ไม่พอใจ', 'รำคาญ'],
            'fear': ['กลัว', 'กังวล', 'ประหม่า', 'วิตก'],
            'surprise': ['ประหลาดใจ', 'ตกใจ', 'อัศจรรย์'],
            'love': ['รัก', 'ห่วง', 'ผูกพัน', 'อบอุ่น'],
            'neutral': ['ปกติ', 'เฉยๆ', 'ธรรมดา']
        }
        
        text_lower = text.lower()
        emotion_counts = {emotion: 0 for emotion in emotion_keywords}
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                emotion_counts[emotion] += text_lower.count(keyword.lower())
                
        # หาอารมณ์ที่พบมากที่สุด
        primary_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        
        return {
            'primary_emotion': primary_emotion[0] if primary_emotion[1] > 0 else 'neutral',
            'emotion_counts': emotion_counts
        }

    def _measure_emotional_intensity(self, text: str) -> Dict:
        """วัดความเข้มข้นของอารมณ์"""
        intensity_markers = {
            'high': ['มาก', 'สุดๆ', 'ที่สุด', 'มากมาย', 'อย่างยิ่ง', '!'],
            'medium': ['ค่อนข้าง', 'พอสมควร', 'ระดับหนึ่ง'],
            'low': ['เล็กน้อย', 'นิดหน่อย', 'บ้าง', 'เล็ก ๆ']
        }
        
        text_lower = text.lower()
        intensity_scores = {level: 0 for level in intensity_markers}
        
        for level, markers in intensity_markers.items():
            for marker in markers:
                count = text_lower.count(marker.lower())
                if level == 'high':
                    intensity_scores[level] += count * 3
                elif level == 'medium':
                    intensity_scores[level] += count * 2
                else:
                    intensity_scores[level] += count
                    
        total_score = sum(intensity_scores.values())
        if total_score == 0:
            return {'level': 'neutral', 'score': 0, 'details': intensity_scores}
            
        # คำนวณระดับความเข้มข้นรวม
        weighted_score = (
            intensity_scores['high'] * 3 +
            intensity_scores['medium'] * 2 +
            intensity_scores['low']
        ) / total_score
        
        if weighted_score >= 2.5:
            level = 'high'
        elif weighted_score >= 1.5:
            level = 'medium'
        else:
            level = 'low'
            
        return {
            'level': level,
            'score': weighted_score,
            'details': intensity_scores
        }

    def _identify_growth_area(self, text: str) -> Dict:
        """ระบุพื้นที่การเติบโต"""
        growth_areas = {
            'knowledge': ['เรียนรู้', 'ความรู้', 'เข้าใจ', 'ศึกษา'],
            'skill': ['ทักษะ', 'ความสามารถ', 'ฝึกฝน', 'พัฒนา'],
            'mindset': ['ทัศนคติ', 'มุมมอง', 'แนวคิด', 'ความเชื่อ'],
            'emotional': ['อารมณ์', 'ความรู้สึก', 'EQ', 'การจัดการอารมณ์'],
            'social': ['สัมพันธภาพ', 'การสื่อสาร', 'ทีม', 'เครือข่าย'],
            'spiritual': ['จิตวิญญาณ', 'ความหมาย', 'เป้าหมายชีวิต']
        }
        
        text_lower = text.lower()
        area_scores = {area: 0 for area in growth_areas}
        
        for area, keywords in growth_areas.items():
            for keyword in keywords:
                count = text_lower.count(keyword.lower())
                area_scores[area] += count
                
        # หาพื้นที่การเติบโตหลักและรอง
        sorted_areas = sorted(area_scores.items(), key=lambda x: x[1], reverse=True)
        primary_area = sorted_areas[0][0] if sorted_areas[0][1] > 0 else 'general'
        secondary_areas = [area for area, score in sorted_areas[1:] if score > 0]
        
        return {
            'primary_area': primary_area,
            'secondary_areas': secondary_areas[:2],  # เก็บแค่ 2 อันดับรอง
            'area_scores': area_scores
        }

    def _find_progress_indicators(self, text: str) -> Dict:
        """ค้นหาตัวบ่งชี้ความก้าวหน้า"""
        progress_indicators = {
            'achievement': ['สำเร็จ', 'บรรลุ', 'ทำได้', 'ผ่าน'],
            'improvement': ['พัฒนา', 'ดีขึ้น', 'ก้าวหน้า', 'เพิ่มขึ้น'],
            'challenge': ['ท้าทาย', 'ปัญหา', 'อุปสรรค', 'ยาก'],
            'learning': ['เรียนรู้', 'เข้าใจ', 'ค้นพบ', 'ได้รู้'],
            'milestone': ['เป้าหมาย', 'ขั้นตอน', 'ระยะ', 'จุดสำคัญ']
        }
        text_lower = text.lower()
        indicator_matches = {category: [] for category in progress_indicators}
        for category, keywords in progress_indicators.items():
            for keyword in keywords:
                if keyword in text_lower:
                    indicator_matches[category].append(keyword)
        has_progress = any(indicator_matches[cat] for cat in indicator_matches)
        indicators_found = [kw for kws in indicator_matches.values() for kw in kws]
        return {
            'has_progress': has_progress,
            'indicators_found': indicators_found,
            'categories': [cat for cat, kws in indicator_matches.items() if kws]
        }

    def _categorize_lesson(self, text: str) -> Dict:
        """จัดหมวดหมู่บทเรียนที่ได้รับจากข้อความ"""
        lesson_categories = {
            'self': ['ตนเอง', 'ข้อผิดพลาด', 'จุดแข็ง', 'จุดอ่อน', 'การยอมรับ'],
            'others': ['ผู้อื่น', 'เพื่อน', 'ครอบครัว', 'ทีม', 'สังคม'],
            'situation': ['สถานการณ์', 'เหตุการณ์', 'ปัญหา', 'โอกาส'],
            'strategy': ['วิธี', 'กลยุทธ์', 'แนวทาง', 'เทคนิค'],
            'emotion': ['อารมณ์', 'ความรู้สึก', 'การจัดการอารมณ์'],
            'general': ['บทเรียน', 'ข้อคิด', 'ประสบการณ์']
        }
        text_lower = text.lower()
        category_scores = {cat: 0 for cat in lesson_categories}
        for cat, keywords in lesson_categories.items():
            for kw in keywords:
                if kw in text_lower:
                    category_scores[cat] += 1
        sorted_cats = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        primary_category = sorted_cats[0][0] if sorted_cats[0][1] > 0 else 'general'
        secondary_categories = [cat for cat, score in sorted_cats[1:] if score > 0]
        return {
            'primary_category': primary_category,
            'secondary_categories': secondary_categories[:2],
            'category_scores': category_scores
        }

    def _assess_wisdom_applicability(self, text: str) -> dict:
        """ประเมินความนำไปใช้ได้ของบทเรียน/ปัญญา"""
        applicability_levels = {
            'high': ['นำไปใช้', 'ประยุกต์', 'ใช้ได้ทันที', 'เหมาะสม', 'สำคัญ'],
            'medium': ['อาจจะใช้', 'ควรลอง', 'น่าสนใจ', 'มีประโยชน์'],
            'low': ['เฉยๆ', 'ไม่แน่ใจ', 'อาจจะ', 'ต้องคิดเพิ่ม']
        }
        text_lower = text.lower()
        scores = {level: 0 for level in applicability_levels}
        for level, keywords in applicability_levels.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[level] += 1
        if scores['high'] > 0:
            level = 'high'
        elif scores['medium'] > 0:
            level = 'medium'
        elif scores['low'] > 0:
            level = 'low'
        else:
            level = 'unknown'
        return {
            'level': level,
            'scores': scores
        }

    def _identify_connection_type(self, text: str) -> dict:
        """ระบุประเภทของการเชื่อมโยงในข้อความ (เช่น เหตุและผล, ความขัดแย้ง, ลำดับ)"""
        connection_types = {
            'cause_effect': ['เพราะ', 'เนื่องจาก', 'ส่งผล', 'ทำให้', 'ผลลัพธ์', 'สาเหตุ'],
            'contrast': ['แต่', 'อย่างไรก็ตาม', 'ในทางกลับกัน', 'ขัดแย้ง', 'ตรงข้าม'],
            'sequence': ['จากนั้น', 'ต่อมา', 'สุดท้าย', 'ก่อน', 'หลังจาก', 'ลำดับ'],
            'general': []
        }
        text_lower = text.lower()
        type_scores = {typ: 0 for typ in connection_types}
        for typ, keywords in connection_types.items():
            for kw in keywords:
                if kw in text_lower:
                    type_scores[typ] += 1
        sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
        primary_type = sorted_types[0][0] if sorted_types[0][1] > 0 else 'general'
        secondary_types = [typ for typ, score in sorted_types[1:] if score > 0]
        return {
            'primary_type': primary_type,
            'secondary_types': secondary_types[:2],
            'type_scores': type_scores
        }

    # --- Chain of Memory Integration ---
    def update_chain_of_memory(self, summary_id: str, summary_file_path: Path, chain_type: str = "summary"):
        """อัปเดต chain of memory ใน NewMemoryManager"""
        if self.new_memory_manager:
            self.new_memory_manager.add_to_chain(summary_id, str(summary_file_path), chain_type=chain_type)

    def get_memory_chain_for_period(self, start_date: datetime, end_date: datetime, chain_type: str = "summary"):
        """ดึง chain of memory สำหรับช่วงเวลาที่กำหนด"""
        if self.new_memory_manager:
            return self.new_memory_manager.get_chain_for_period(start_date, end_date, chain_type=chain_type)
        return []

    def merge_memory_chains(self, chain_ids: list, target_chain_id: str, chain_type: str = "summary"):
        """รวม chain หลายอันเข้าด้วยกัน"""
        if self.new_memory_manager:
            return self.new_memory_manager.merge_chains(chain_ids, target_chain_id, chain_type=chain_type)
        return False

    # --- IdentityCore Integration ---
    def get_identity_context(self) -> dict:
        """ดึง identity context จาก IdentityCore ถ้ามี (สำหรับ enrich reflection/summarization)"""
        if self.identity_core and hasattr(self.identity_core, 'get_identity_context'):
            try:
                return self.identity_core.get_identity_context()
            except Exception as e:
                logger.warning(f"Error fetching identity context: {e}")
        return {}

    def fetch_memories_for_summary(self, start_date: datetime, end_date: datetime, filter_tags: list = None) -> list:
        """ดึง atomic memories จาก NewMemoryManager สำหรับการสรุป/ผสาน"""
        if self.new_memory_manager and hasattr(self.new_memory_manager, 'get_memories_for_period'):
            try:
                return self.new_memory_manager.get_memories_for_period(start_date, end_date, filter_tags=filter_tags)
            except Exception as e:
                logger.warning(f"Error fetching memories from NewMemoryManager: {e}")
        return []

    def update_chain_of_memory(self, summary_id: str, summary_file_path: Path, chain_type: str = "summary"):
        """อัพเดท chain-of-memory ทั้งใน summary และ memory object (ถ้ามี NewMemoryManager)"""
        # Update summary file (already handled in _update_source_summary_chains)
        # Update memory objects if possible
        if self.new_memory_manager and hasattr(self.new_memory_manager, 'update_memory_chain'):
            try:
                self.new_memory_manager.update_memory_chain(summary_id=summary_id, summary_file_path=summary_file_path, chain_type=chain_type)
            except Exception as e:
                logger.warning(f"Error updating memory chain in NewMemoryManager: {e}")

    def on_new_raw_chat_log(file_path, rel_path, metadata, text, timestamp):
        """
        Callback สำหรับ raw_chat_logger: สร้าง summary จาก raw chat ใหม่
        """
        try:
            # สร้าง PathManager และ EthernityMemorySummarizer instance
            from core.path_manager import PathManager
            path_manager = PathManager(root_dir="memory_core")
            summarizer = EthernityMemorySummarizer(path_manager_instance=path_manager)
            # เรียกฟังก์ชันสรุป (สมมุติชื่อ summarize_new_chat, ต้องปรับตามจริง)
            if hasattr(summarizer, "summarize_new_chat"):
                summarizer.summarize_new_chat(file_path=file_path, rel_path=rel_path, metadata=metadata, text=text, timestamp=timestamp)
            else:
                print("[EthernityMemorySummarizer] summarize_new_chat() not implemented.")
        except Exception as e:
            print(f"[EthernityMemorySummarizer] Failed to summarize raw chat: {e}")