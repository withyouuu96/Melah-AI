# core/mcts_engine.py

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import math

# เพื่อให้สามารถ type hint ได้โดยไม่เกิด Circular Import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .identity_core import IdentityCore
    from .vector_memory_index import VectorMemoryIndex
    from .memory_path_bridge import MemoryPathBridge
    from .reflector import Reflector # For reflection/learning
    # from .llm_connector import LLMConnector # เข้าถึงผ่าน IdentityCore


logger = logging.getLogger(__name__)

class MCTS_Engine:
    """
    ใช้ Monte Carlo Tree Search (MCTS) พร้อมกลไก Reflective (R-MCTS)
    เพื่อสำรวจและดึง "เส้นทาง" ของความทรงจำที่เกี่ยวข้องมากที่สุด
    """
    def __init__(self, identity_core_instance: 'IdentityCore',
                 vector_memory_index: 'VectorMemoryIndex',
                 memory_path_bridge: 'MemoryPathBridge',
                 config: Optional[Dict] = None):
        
        self.identity_core = identity_core_instance
        self.vector_memory_index = vector_memory_index
        self.memory_path_bridge = memory_path_bridge
        
        # การตั้งค่าสำหรับ MCTS (สามารถปรับแต่งได้)
        self.config = {
            "max_simulations": 50,          # จำนวนครั้งสูงสุดในการจำลอง
            "max_depth": 5,                 # ความลึกสูงสุดของ chain ความจำที่ค้นหา
            "uct_c": 1.41,                  # ค่าคงที่สำหรับ UCT (Upper Confidence Bound 1 applied to Trees)
            "exploration_weight": 0.7,      # น้ำหนักสำหรับการสำรวจ (exploration) เทียบกับการใช้ประโยชน์ (exploitation)
            "llm_evaluation_threshold": 0.6 # คะแนนที่ LLM ต้องประเมินได้ถึง (ถ้าใช้ LLM ใน simulation)
        }
        if config:
            self.config.update(config)

        self.root_node = None # โหนดเริ่มต้นของ MCTS Tree

        logger.info("🌳 MCTS_Engine initialized.")

    def search_memory_chain(self, query: str, exclude_session_ids: Optional[set] = None) -> Dict[str, Any]:
        """
        เริ่มกระบวนการ R-MCTS เพื่อค้นหา chain ความทรงจำที่ดีที่สุด
        
        Args:
            query (str): คำถามของผู้ใช้
            exclude_session_ids (Optional[set]): เซสชันที่เคยล้มเหลวหรือไม่ต้องการ
        
        Returns:
            Dict: {'context': str, 'session_id': str, 'score': float}
        """
        if not self.vector_memory_index.embeddings:
            logger.warning("VectorMemoryIndex not ready. Falling back to basic memory retrieval.")
            # อาจจะเรียก IdentityCore.get_memory_context_for_query แบบเดิมแทน
            return self.identity_core.get_memory_context_for_query(query, exclude_session_ids=exclude_session_ids)

        logger.info(f"Starting MCTS search for query: '{query[:50]}...'")
        
        # Initialize the root node
        # Root node แทนสถานะเริ่มต้น ที่ยังไม่มีความจำถูกดึงมา
        self.root_node = MCTSNode(
            session_id="ROOT",
            parent=None,
            depth=0,
            query=query,
            exclude_ids=exclude_session_ids or set()
        )
        
        best_context = ""
        best_session_id = None
        best_score = -1.0 # คะแนนของ chain ที่ดีที่สุด
        
        for i in range(self.config["max_simulations"]):
            logger.debug(f"MCTS Simulation {i+1}/{self.config['max_simulations']}")
            
            # 1. Selection: เลือกโหนดที่จะสำรวจ (traverse tree)
            selected_node = self._select(self.root_node)
            
            # 2. Expansion: ขยายโหนดที่เลือก (add new child nodes)
            # ถ้าโหนดนี้ยังไม่เคยขยาย หรือยังไม่ถึง max_depth
            if not selected_node.is_terminal and selected_node.depth < self.config["max_depth"]:
                self._expand(selected_node)
            
            # 3. Simulation (Rollout): จำลองการเล่นไปจนจบ (จาก selected_node)
            # เราจะจำลองการดึงความจำจาก selected_node และประเมินผลลัพธ์
            rollout_score, rollout_context, rollout_session_id = self._simulate(selected_node)
            
            # 4. Backpropagation: อัปเดตสถิติย้อนกลับขึ้นไปบน Tree
            self._backpropagate(selected_node, rollout_score)
            
            # อัปเดตผลลัพธ์ที่ดีที่สุดที่เจอระหว่าง simulation
            if rollout_score > best_score:
                best_score = rollout_score
                best_context = rollout_context
                best_session_id = rollout_session_id
                logger.debug(f"New best path found (score: {best_score:.4f}) ending at session: {best_session_id}")
        
        logger.info(f"MCTS search finished. Best score: {best_score:.4f}, Session: {best_session_id}")
        return {"context": best_context, "session_id": best_session_id, "score": best_score}

    def _select(self, node: 'MCTSNode') -> 'MCTSNode':
        """
        เลือกโหนดที่ดีที่สุดเพื่อสำรวจต่อไป (ใช้ UCT)
        """
        while not node.is_terminal and node.children:
            best_child = None
            best_uct_value = -float('inf')

            for child in node.children:
                if child.visits == 0: # ถ้ายังไม่เคยไปโหนดนี้ ให้เลือกก่อนเพื่อสำรวจ
                    return child
                
                # UCT formula: Q/N + C * sqrt(ln(N_parent) / N_child)
                # Q = total_value (ความสำเร็จ)
                # N = visits (จำนวนครั้งที่เข้าชม)
                exploit_term = child.total_value / child.visits
                explore_term = self.config["uct_c"] * math.sqrt(math.log(node.visits) / child.visits)
                
                uct_value = exploit_term + explore_term
                
                if uct_value > best_uct_value:
                    best_uct_value = uct_value
                    best_child = child
            node = best_child
        return node

    def _expand(self, node: 'MCTSNode'):
        """
        ขยายโหนดโดยการหา session_id ที่เกี่ยวข้องและสร้าง child nodes
        """
        # ใช้ VectorMemoryIndex เพื่อหา session ที่เกี่ยวข้องกับ query ของ node
        # (หรือ query หลักของ root node)
        # ต้องระวังไม่ให้ดึง session ที่ซ้ำกับ ancestor หรือที่ถูก exclude_ids
        
        # สำหรับโหนด ROOT, เราจะใช้ query หลัก
        # สำหรับโหนดอื่นๆ เราอาจจะใช้บริบทที่รวบรวมได้จากเส้นทางถึงโหนดนั้น
        # เพื่อหา session ที่เกี่ยวข้องถัดไป (Semantic Chaining)
        
        current_context_for_search = node.query # เบื้องต้นใช้ query หลักไปก่อน
        # ถ้าอยากให้ชาญฉลาดขึ้น: อาจจะดึง content ของ node.session_id มา
        # แล้วใช้ nlp_processor.summarize_text หรือ LLM เพื่อสร้าง query ใหม่สำหรับ expansion
        
        # ดึง session ที่เกี่ยวข้อง (ไม่รวมที่เคยถูก exclude หรือที่อยู่ใน path ปัจจุบัน)
        exclude_ids_for_search = node.exclude_ids.union(node.get_path_session_ids())
        
        relevant_sessions = self.vector_memory_index.search(
            query=current_context_for_search,
            top_k=5, # ลองหา top N sessions ที่เกี่ยวข้อง
            exclude_ids=list(exclude_ids_for_search)
        )
        
        if not relevant_sessions:
            node.is_terminal = True # ไม่มีอะไรให้ขยาย
            return

        for result in relevant_sessions:
            session_id = result['session_id']
            # สร้าง child node สำหรับแต่ละ session ที่พบ
            new_child = MCTSNode(
                session_id=session_id,
                parent=node,
                depth=node.depth + 1,
                query=self.root_node.query, # child node ยังคงอ้างอิง query หลัก
                exclude_ids=exclude_ids_for_search # ส่ง exclude_ids ไปให้ child node ด้วย
            )
            node.add_child(new_child)
            logger.debug(f"Expanded node {node.session_id} with child: {session_id}")

    def _simulate(self, node: 'MCTSNode') -> Tuple[float, str, str]:
        """
        จำลองการดึงความจำจากโหนดที่เลือก และประเมินคุณภาพ
        นี่คือจุดที่อาจจะมีการเรียก LLM (ผ่าน IdentityCore)
        """
        current_session_id = node.session_id
        current_context = ""
        
        # ดึงเนื้อหาของ chain ความจำที่นำมาถึงโหนดนี้
        # หรืออาจจะแค่ดึงเนื้อหาของ current_session_id แล้วรวมกับ context จาก parent
        
        # ในการจำลอง เราจะ "สร้าง" chain ความจำขึ้นมา
        # โดยดึง content จาก session_id ของ node และ ancestor ของมัน
        
        # Get the path from root to current node
        path_session_ids = node.get_path_session_ids()
        
        # ดึงเนื้อหาของแต่ละ session ใน path
        # Note: self.memory_path_bridge.get_sessions_content_from_meta อาจจะเหมาะกว่า
        # แต่จากไฟล์ memory_manager.py, คุณมี get_chain_context ที่รับ summary_filename
        # อาจจะต้องปรับ memory_path_bridge ให้ดึง content ตาม session_id list ได้ง่ายขึ้น
        
        # สมมติว่าเรามี method ที่สามารถดึงเนื้อหาจาก list ของ session_ids ได้
        # ถ้า MemoryPathBridge.get_sessions_content (ที่รับ List[str] ของ full paths) ถูกใช้,
        # เราอาจจะต้องหา full path ของแต่ละ session_id
        
        # สำหรับความง่ายในการร่างเบื้องต้น: ดึงเนื้อหาแค่ session_id ของ node
        # และใช้ self.memory_path_bridge.get_sessions_content_from_meta(session_id)
        # เพื่อดึง chain ที่เหลือ
        
        simulated_chain_contents = []
        if current_session_id != "ROOT": # ไม่ต้องดึงเนื้อหาถ้าเป็น root node
            # ดึงเนื้อหาของ session นี้
            # Note: ต้องระบุ path ให้ถูก หรือปรับ MemoryManager ให้รับ session_id ตรงๆ
            # สมมติว่า memory_path_bridge.get_session_content(session_id) มีอยู่
            # หรือใช้ get_sessions_content_from_meta แล้วส่ง list แค่ session_id เดียว
            
            # ดึง chain ที่นำมาถึง node นี้
            session_meta_chain = self.memory_path_bridge.get_session_chain(current_session_id, max_depth=node.depth + 1)
            raw_contents = self.memory_path_bridge.get_sessions_content_from_meta(session_meta_chain)
            
            if raw_contents:
                # สังเคราะห์บริบทจากเนื้อหา
                # นี่คือจุดที่อาจจะใช้ LLM หรือ NLPProcessor เพื่อสรุป/เรียบเรียงบริบท
                combined_content = "\n---\n".join(raw_contents)
                if len(combined_content) > 1000: # ถ้าบริบทใหญ่เกินไป อาจจะสรุป
                    try:
                        # LLM Call (ผ่าน IdentityCore) สำหรับการประเมินบริบท
                        llm_client = self.identity_core.llm
                        eval_prompt = f"Summarize the following conversation context briefly, focusing on key information related to the query '{node.query}':\n\n{combined_content}"
                        current_context = llm_client.generate(prompt=eval_prompt, context="")
                    except Exception as e:
                        logger.warning(f"LLM evaluation in MCTS simulation failed: {e}. Using raw combined content.")
                        current_context = combined_content
                else:
                    current_context = combined_content
            
        # การประเมินคะแนน (Rollout Score)
        # นี่คือจุดสำคัญที่ R-MCTS จะประเมินว่าบริบทที่ได้มีคุณภาพแค่ไหน
        score = self._evaluate_context_quality(node.query, current_context)

        # หากเป็นการจำลองครั้งแรกๆ อาจจะส่งคะแนนเฉลี่ย
        # หรือถ้าเจอความสำเร็จ ให้คะแนนสูง
        if score is None: # ถ้าไม่มีการประเมินที่ชัดเจน ให้คะแนนเริ่มต้น
            score = 0.5 # Default neutral score

        return score, current_context, current_session_id

    def _evaluate_context_quality(self, query: str, context: str) -> float:
        """
        ประเมินคุณภาพของบริบทที่ดึงมาว่าเกี่ยวข้องกับ query แค่ไหน
        อาจใช้ LLM (ผ่าน IdentityCore) หรือ Vector Search อีกครั้งเพื่อประเมิน
        """
        if not context.strip():
            return 0.0 # บริบทว่างเปล่า ให้คะแนน 0

        # Option 1: ใช้ LLM เพื่อประเมิน
        try:
            llm_client = self.identity_core.llm
            eval_prompt = f"""Given the user's query: "{query}" and the retrieved context: "{context}", how relevant and useful is this context for answering the query?
            Provide a relevance score from 0.0 (not relevant) to 1.0 (highly relevant and useful).
            Output ONLY the score as a float.
            """
            llm_response = llm_client.generate(prompt=eval_prompt, context="")
            score = float(llm_response.strip())
            # ใช้ Reflector เพื่อปรับปรุงการเรียนรู้ (Contrastive Reflection)
            # ถ้า score ต่ำมากๆ, อาจจะส่งข้อมูลนี้ให้ reflector เพื่อพิจารณาว่าทำไมถึงไม่ดี
            if score < 0.3:
                self.identity_core.reflector.llm_reflect(
                    raw_thought=f"Low relevance context for query: '{query}' from session: {node.session_id}",
                    conversation_context=[], # อาจจะส่ง context ที่เกี่ยวข้อง
                    long_term_memory=context,
                    used_session_id=node.session_id,
                    current_intention=self.identity_core.current_seed.get("intention", "Default: Be a helpful assistant.")
                )
            return score
        except Exception as e:
            logger.warning(f"LLM evaluation for context quality failed: {e}. Falling back to semantic similarity.")
            # Fallback to Option 2: ใช้ Semantic Similarity (Vector Search)
            return self._semantic_similarity_score(query, context)

    def _semantic_similarity_score(self, query: str, context: str) -> float:
        """
        ประเมินคะแนนความคล้ายคลึงเชิงความหมายระหว่าง query และ context
        """
        if not self.vector_memory_index.model:
            return 0.0 # ถ้าโมเดลไม่พร้อม

        try:
            query_embedding = self.vector_memory_index.model.encode(query, convert_to_tensor=True)
            context_embedding = self.vector_memory_index.model.encode(context, convert_to_tensor=True)
            score = float(np.array(self.vector_memory_index.util.cos_sim(query_embedding, context_embedding))[0][0])
            return score
        except Exception as e:
            logger.error(f"Semantic similarity scoring failed: {e}")
            return 0.0

    def _backpropagate(self, node: 'MCTSNode', score: float):
        """
        อัปเดตสถิติของโหนดบนเส้นทางจากโหนดที่เลือกไปจนถึง root
        """
        while node is not None:
            node.visits += 1
            node.total_value += score
            node = node.parent


class MCTSNode:
    """
    โหนดใน MCTS Tree แทนสถานะหนึ่งๆ (เช่น chain ความจำที่ได้)
    """
    def __init__(self, session_id: str, parent: Optional['MCTSNode'], depth: int, query: str, exclude_ids: set):
        self.session_id = session_id
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits = 0 # จำนวนครั้งที่โหนดนี้ถูกเข้าชม
        self.total_value = 0.0 # ผลรวมของคะแนนความสำเร็จจากการจำลองที่ผ่านโหนดนี้
        self.depth = depth
        self.is_terminal = False # เป็นโหนดสุดท้ายที่ขยายไม่ได้แล้ว
        self.query = query # เก็บ query หลักไว้ในโหนดเพื่อใช้ในการประเมิน
        self.exclude_ids = exclude_ids # session IDs ที่ไม่ต้องการนำมาใช้ในเส้นทางนี้

    def add_child(self, child_node: 'MCTSNode'):
        self.children.append(child_node)

    def get_path_session_ids(self) -> List[str]:
        """คืนค่า list ของ session_ids จาก root จนถึงโหนดปัจจุบัน"""
        path = []
        current = self
        while current and current.session_id != "ROOT":
            path.insert(0, current.session_id)
            current = current.parent
        return path

    def __repr__(self):
        return f"MCTSNode(id={self.session_id}, visits={self.visits}, value={self.total_value:.2f}, depth={self.depth})"