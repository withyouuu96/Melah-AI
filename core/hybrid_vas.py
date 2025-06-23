import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import statistics

class RiskLevel(Enum):
    """Risk levels for different contexts"""
    CRITICAL = "critical"        # Medical, financial, safety-critical
    HIGH = "high"               # Legal, security, important business
    MEDIUM = "medium"           # General business, education
    LOW = "low"                 # Creative, personal, experimental

class ContextType(Enum):
    """Different contexts for VAS application"""
    MEDICAL = "medical"
    FINANCIAL = "financial"
    LEGAL = "legal"
    SAFETY_CRITICAL = "safety_critical"
    BUSINESS = "business"
    EDUCATION = "education"
    CREATIVE = "creative"
    PERSONAL = "personal"
    DEVELOPMENT = "development"
    RESEARCH = "research"

class HybridVAS:
    """
    Hybrid Value/Affective System that adapts behavior based on context and risk level
    
    - Critical contexts: Evidence-based decisions only
    - Medium contexts: VAS-guided with safety checks
    - Low contexts: Full VAS with experiential learning
    """
    
    def __init__(self):
        """Initialize Hybrid VAS with context-aware configurations"""
        
        # Context-specific configurations
        self.context_configs = {
            ContextType.MEDICAL: {
                'risk_level': RiskLevel.CRITICAL,
                'evidence_weight': 1.0,
                'experience_weight': 0.0,
                'safety_threshold': 0.95,
                'requires_validation': True,
                'allows_experimentation': False
            },
            ContextType.FINANCIAL: {
                'risk_level': RiskLevel.CRITICAL,
                'evidence_weight': 0.9,
                'experience_weight': 0.1,
                'safety_threshold': 0.9,
                'requires_validation': True,
                'allows_experimentation': False
            },
            ContextType.LEGAL: {
                'risk_level': RiskLevel.HIGH,
                'evidence_weight': 0.8,
                'experience_weight': 0.2,
                'safety_threshold': 0.85,
                'requires_validation': True,
                'allows_experimentation': False
            },
            ContextType.BUSINESS: {
                'risk_level': RiskLevel.MEDIUM,
                'evidence_weight': 0.6,
                'experience_weight': 0.4,
                'safety_threshold': 0.7,
                'requires_validation': False,
                'allows_experimentation': True
            },
            ContextType.EDUCATION: {
                'risk_level': RiskLevel.MEDIUM,
                'evidence_weight': 0.5,
                'experience_weight': 0.5,
                'safety_threshold': 0.6,
                'requires_validation': False,
                'allows_experimentation': True
            },
            ContextType.DEVELOPMENT: {
                'risk_level': RiskLevel.LOW,
                'evidence_weight': 0.3,
                'experience_weight': 0.7,
                'safety_threshold': 0.5,
                'requires_validation': False,
                'allows_experimentation': True
            },
            ContextType.CREATIVE: {
                'risk_level': RiskLevel.LOW,
                'evidence_weight': 0.2,
                'experience_weight': 0.8,
                'safety_threshold': 0.3,
                'requires_validation': False,
                'allows_experimentation': True
            }
        }
        
        # Separate storage for different contexts
        self.evidence_based_decisions = []
        self.experience_based_decisions = []
        self.hybrid_decisions = []
        
        # Context-specific mindsets
        self.mindsets = {
            context: {
                'certainty_preference': 0.8 if config['risk_level'] in [RiskLevel.CRITICAL, RiskLevel.HIGH] else 0.5,
                'innovation_tolerance': 0.2 if config['risk_level'] == RiskLevel.CRITICAL else 0.7,
                'experience_trust': config['experience_weight'],
                'validation_strictness': 0.9 if config['requires_validation'] else 0.5
            }
            for context, config in self.context_configs.items()
        }
        
        # Performance tracking per context
        self.performance_metrics = {
            context: {
                'total_decisions': 0,
                'successful_outcomes': 0,
                'accuracy': 0.0,
                'safety_violations': 0
            }
            for context in ContextType
        }
    
    def make_decision(self,
                     context: ContextType,
                     decision_request: Dict,
                     evidence: List[str] = None,
                     past_experiences: List[Dict] = None,
                     user_preferences: Dict = None) -> Dict:
        """
        Make context-aware decision using appropriate VAS strategy
        
        Args:
            context: The context type for this decision
            decision_request: Details about the decision needed
            evidence: Available evidence/data
            past_experiences: Relevant past experiences
            user_preferences: User's preferences and constraints
            
        Returns:
            Decision with reasoning and confidence level
        """
        
        config = self.context_configs[context]
        mindset = self.mindsets[context]
        
        # Select decision strategy based on context
        if config['risk_level'] == RiskLevel.CRITICAL:
            return self._make_evidence_based_decision(
                context, decision_request, evidence, config
            )
        elif config['risk_level'] == RiskLevel.HIGH:
            return self._make_hybrid_decision(
                context, decision_request, evidence, past_experiences, config
            )
        else:  # MEDIUM or LOW risk
            return self._make_experience_enhanced_decision(
                context, decision_request, evidence, past_experiences, user_preferences, config
            )
    
    def _make_evidence_based_decision(self,
                                    context: ContextType,
                                    request: Dict,
                                    evidence: List[str],
                                    config: Dict) -> Dict:
        """
        Make purely evidence-based decision for critical contexts
        """
        
        decision = {
            'context': context.value,
            'strategy': 'evidence_based',
            'timestamp': datetime.now().isoformat(),
            'request': request,
            'evidence_quality': self._assess_evidence_quality(evidence or []),
            'confidence': 0.0,
            'recommendation': None,
            'reasoning': [],
            'safety_checks': [],
            'requires_human_review': config['requires_validation']
        }
        
        # Assess evidence quality
        if not evidence:
            decision['confidence'] = 0.1
            decision['recommendation'] = "INSUFFICIENT_EVIDENCE"
            decision['reasoning'] = ["No evidence provided - cannot make safe decision"]
            decision['safety_checks'] = ["⚠️ CRITICAL: Evidence required for this context"]
        else:
            evidence_score = self._assess_evidence_quality(evidence)
            decision['confidence'] = evidence_score
            
            if evidence_score >= config['safety_threshold']:
                decision['recommendation'] = self._generate_evidence_based_recommendation(evidence, request)
                decision['reasoning'] = [f"Decision based on {len(evidence)} evidence items"]
                decision['safety_checks'] = ["✅ Evidence quality meets safety threshold"]
            else:
                decision['recommendation'] = "SEEK_MORE_EVIDENCE"
                decision['reasoning'] = ["Evidence quality below safety threshold"]
                decision['safety_checks'] = ["⚠️ Additional evidence required"]
        
        self.evidence_based_decisions.append(decision)
        return decision
    
    def _make_hybrid_decision(self,
                            context: ContextType,
                            request: Dict,
                            evidence: List[str],
                            experiences: List[Dict],
                            config: Dict) -> Dict:
        """
        Make hybrid decision combining evidence and experience
        """
        
        decision = {
            'context': context.value,
            'strategy': 'hybrid',
            'timestamp': datetime.now().isoformat(),
            'request': request,
            'evidence_weight': config['evidence_weight'],
            'experience_weight': config['experience_weight'],
            'confidence': 0.0,
            'recommendation': None,
            'reasoning': [],
            'safety_checks': []
        }
        
        # Calculate evidence component
        evidence_score = self._assess_evidence_quality(evidence or []) * config['evidence_weight']
        
        # Calculate experience component
        experience_score = 0.0
        if experiences and config['allows_experimentation']:
            experience_score = self._assess_experience_relevance(experiences, request) * config['experience_weight']
        
        # Combined confidence
        combined_confidence = evidence_score + experience_score
        decision['confidence'] = combined_confidence
        
        # Safety check
        if combined_confidence >= config['safety_threshold']:
            decision['recommendation'] = self._generate_hybrid_recommendation(
                evidence, experiences, request, config
            )
            decision['reasoning'] = [
                f"Evidence contribution: {evidence_score:.2f}",
                f"Experience contribution: {experience_score:.2f}",
                f"Combined confidence: {combined_confidence:.2f}"
            ]
            decision['safety_checks'] = ["✅ Combined confidence meets threshold"]
        else:
            decision['recommendation'] = "REQUIRES_REVIEW"
            decision['reasoning'] = ["Combined confidence below safety threshold"]
            decision['safety_checks'] = ["⚠️ Human review recommended"]
        
        self.hybrid_decisions.append(decision)
        return decision
    
    def _make_experience_enhanced_decision(self,
                                         context: ContextType,
                                         request: Dict,
                                         evidence: List[str],
                                         experiences: List[Dict],
                                         preferences: Dict,
                                         config: Dict) -> Dict:
        """
        Make experience-enhanced decision for low-risk contexts
        """
        
        decision = {
            'context': context.value,
            'strategy': 'experience_enhanced',
            'timestamp': datetime.now().isoformat(),
            'request': request,
            'vas_scores': {},
            'confidence': 0.0,
            'recommendation': None,
            'reasoning': [],
            'learning_opportunity': True
        }
        
        # Apply full VAS evaluation
        if experiences:
            vas_evaluation = self._apply_full_vas(experiences, request, preferences or {})
            decision['vas_scores'] = vas_evaluation
            decision['confidence'] = vas_evaluation['overall_score']
        else:
            decision['confidence'] = 0.5  # Neutral confidence without experience
        
        # Generate recommendation
        decision['recommendation'] = self._generate_experience_based_recommendation(
            evidence, experiences, preferences, request
        )
        
        decision['reasoning'] = [
            "Experience-based decision appropriate for this context",
            f"VAS confidence: {decision['confidence']:.2f}",
            "Learning from outcome will improve future decisions"
        ]
        
        self.experience_based_decisions.append(decision)
        return decision
    
    def _assess_evidence_quality(self, evidence: List[str]) -> float:
        """Assess quality of evidence provided"""
        if not evidence:
            return 0.0
        
        quality_indicators = ['peer_reviewed', 'data', 'study', 'research', 'validated']
        quality_score = 0.0
        
        for item in evidence:
            item_lower = item.lower()
            score = 0.3  # Base score
            
            for indicator in quality_indicators:
                if indicator in item_lower:
                    score += 0.15
            
            quality_score += min(score, 1.0)
        
        return min(quality_score / len(evidence), 1.0)
    
    def _assess_experience_relevance(self, experiences: List[Dict], request: Dict) -> float:
        """Assess relevance of past experiences to current request"""
        if not experiences:
            return 0.0
        
        relevance_scores = []
        for exp in experiences:
            # Simple relevance calculation based on keywords match
            exp_text = str(exp).lower()
            request_text = str(request).lower()
            
            common_words = set(exp_text.split()) & set(request_text.split())
            relevance = len(common_words) / max(len(request_text.split()), 1)
            relevance_scores.append(min(relevance, 1.0))
        
        return statistics.mean(relevance_scores) if relevance_scores else 0.0
    
    def _apply_full_vas(self, experiences: List[Dict], request: Dict, preferences: Dict) -> Dict:
        """Apply full VAS evaluation for low-risk contexts"""
        
        # Calculate VAS factors
        emotion_score = preferences.get('emotional_weight', 0.5)
        significance_score = 0.7  # Default significance
        novelty_score = 0.6 if 'new' in str(request).lower() else 0.4
        long_term_score = preferences.get('long_term_focus', 0.5)
        
        overall_score = (emotion_score + significance_score + novelty_score + long_term_score) / 4
        
        return {
            'emotion': emotion_score,
            'significance': significance_score,
            'novelty': novelty_score,
            'long_term_impact': long_term_score,
            'overall_score': overall_score
        }
    
    def _generate_evidence_based_recommendation(self, evidence: List[str], request: Dict) -> str:
        """Generate recommendation based purely on evidence"""
        if not evidence:
            return "Cannot provide recommendation without evidence"
        
        return f"Based on {len(evidence)} evidence items: Proceed with evidence-supported approach"
    
    def _generate_hybrid_recommendation(self, 
                                      evidence: List[str], 
                                      experiences: List[Dict], 
                                      request: Dict, 
                                      config: Dict) -> str:
        """Generate hybrid recommendation"""
        return f"Hybrid recommendation: Evidence + Experience guided approach (Weights: {config['evidence_weight']:.1f}/{config['experience_weight']:.1f})"
    
    def _generate_experience_based_recommendation(self,
                                                evidence: List[str],
                                                experiences: List[Dict],
                                                preferences: Dict,
                                                request: Dict) -> str:
        """Generate experience-enhanced recommendation"""
        return "Experience-enhanced recommendation: Leveraging past learning for optimal outcome"
    
    def update_performance(self, 
                          context: ContextType, 
                          decision_id: str, 
                          outcome: str, 
                          success: bool):
        """Update performance metrics based on decision outcomes"""
        
        metrics = self.performance_metrics[context]
        metrics['total_decisions'] += 1
        
        if success:
            metrics['successful_outcomes'] += 1
        
        metrics['accuracy'] = metrics['successful_outcomes'] / metrics['total_decisions']
        
        # Check for safety violations
        if 'violation' in outcome.lower() or 'error' in outcome.lower():
            metrics['safety_violations'] += 1
    
    def get_context_recommendations(self) -> Dict:
        """Get recommendations for when to use each context"""
        
        return {
            'usage_guidelines': {
                ContextType.MEDICAL.value: {
                    'use_for': ['Diagnosis support', 'Treatment planning', 'Drug interactions'],
                    'avoid_for': ['Direct patient care', 'Emergency decisions'],
                    'requirements': ['Medical professional oversight', 'Evidence validation']
                },
                ContextType.FINANCIAL.value: {
                    'use_for': ['Investment analysis', 'Risk assessment', 'Portfolio optimization'],
                    'avoid_for': ['High-stakes trading', 'Regulatory compliance'],
                    'requirements': ['Financial expert review', 'Audit trail']
                },
                ContextType.DEVELOPMENT.value: {
                    'use_for': ['Code architecture', 'Technology selection', 'Performance optimization'],
                    'avoid_for': ['Security-critical code', 'Production deployments'],
                    'requirements': ['Code review', 'Testing protocols']
                },
                ContextType.CREATIVE.value: {
                    'use_for': ['Content generation', 'Design ideas', 'Innovation brainstorming'],
                    'avoid_for': ['Brand-critical content', 'Legal documents'],
                    'requirements': ['Creative review', 'Brand alignment check']
                }
            },
            'risk_mitigation': {
                'always_required': ['Human oversight', 'Outcome tracking', 'Performance monitoring'],
                'context_specific': {
                    'critical': ['Multiple validation', 'Expert review', 'Audit trail'],
                    'high': ['Validation', 'Review process', 'Documentation'],
                    'medium': ['Basic review', 'Performance tracking'],
                    'low': ['Outcome monitoring', 'Learning integration']
                }
            }
        }
    
    def export_system_state(self) -> str:
        """Export complete system state"""
        state = {
            'configurations': {ctx.value: cfg for ctx, cfg in self.context_configs.items()},
            'mindsets': {ctx.value: ms for ctx, ms in self.mindsets.items()},
            'performance_metrics': {ctx.value: pm for ctx, pm in self.performance_metrics.items()},
            'decision_counts': {
                'evidence_based': len(self.evidence_based_decisions),
                'hybrid': len(self.hybrid_decisions),
                'experience_enhanced': len(self.experience_based_decisions)
            },
            'export_timestamp': datetime.now().isoformat()
        }
        return json.dumps(state, indent=2)

# Example usage for different contexts
if __name__ == "__main__":
    # Create hybrid VAS
    hybrid_vas = HybridVAS()
    
    print("=== HYBRID VAS SYSTEM DEMONSTRATION ===\n")
    
    # Example 1: Medical context (Critical - Evidence only)
    print("1. MEDICAL CONTEXT (Critical Risk)")
    medical_decision = hybrid_vas.make_decision(
        context=ContextType.MEDICAL,
        decision_request={'type': 'diagnosis_support', 'symptoms': ['fever', 'cough']},
        evidence=['peer_reviewed study on symptom correlation', 'clinical guidelines for diagnosis']
    )
    print(f"Strategy: {medical_decision['strategy']}")
    print(f"Confidence: {medical_decision['confidence']:.2f}")
    print(f"Recommendation: {medical_decision['recommendation']}")
    print(f"Safety Checks: {medical_decision['safety_checks']}")
    print()
    
    # Example 2: Business context (Medium - Hybrid)
    print("2. BUSINESS CONTEXT (Medium Risk)")
    business_decision = hybrid_vas.make_decision(
        context=ContextType.BUSINESS,
        decision_request={'type': 'product_feature', 'goal': 'increase user engagement'},
        evidence=['market research data', 'competitor analysis'],
        past_experiences=[
            {'feature': 'chat', 'outcome': 'successful', 'user_feedback': 'positive'},
            {'feature': 'notifications', 'outcome': 'mixed', 'user_feedback': 'moderate'}
        ]
    )
    print(f"Strategy: {business_decision['strategy']}")
    print(f"Confidence: {business_decision['confidence']:.2f}")
    print(f"Evidence Weight: {business_decision['evidence_weight']}")
    print(f"Experience Weight: {business_decision['experience_weight']}")
    print()
    
    # Example 3: Creative context (Low - Experience enhanced)
    print("3. CREATIVE CONTEXT (Low Risk)")
    creative_decision = hybrid_vas.make_decision(
        context=ContextType.CREATIVE,
        decision_request={'type': 'content_creation', 'theme': 'technology innovation'},
        evidence=['industry trends', 'audience preferences'],
        past_experiences=[
            {'content_type': 'blog', 'engagement': 'high', 'topic': 'AI'},
            {'content_type': 'video', 'engagement': 'medium', 'topic': 'tech'}
        ],
        user_preferences={'emotional_weight': 0.7, 'long_term_focus': 0.6}
    )
    print(f"Strategy: {creative_decision['strategy']}")
    print(f"Confidence: {creative_decision['confidence']:.2f}")
    print(f"VAS Scores: {creative_decision.get('vas_scores', 'N/A')}")
    print(f"Learning Opportunity: {creative_decision['learning_opportunity']}")
    print()
    
    # Example 4: Development context
    print("4. DEVELOPMENT CONTEXT (Low Risk)")
    dev_decision = hybrid_vas.make_decision(
        context=ContextType.DEVELOPMENT,
        decision_request={'type': 'architecture_choice', 'system': 'web_application'},
        evidence=['performance benchmarks', 'scalability studies'],
        past_experiences=[
            {'framework': 'React', 'outcome': 'successful', 'maintenance': 'easy'},
            {'framework': 'Vue', 'outcome': 'successful', 'learning_curve': 'steep'}
        ],
        user_preferences={'innovation_tolerance': 0.8}
    )
    print(f"Strategy: {dev_decision['strategy']}")
    print(f"Confidence: {dev_decision['confidence']:.2f}")
    print()
    
    # Get usage recommendations
    print("5. USAGE GUIDELINES")
    recommendations = hybrid_vas.get_context_recommendations()
    print("Medical Context Guidelines:")
    medical_guidelines = recommendations['usage_guidelines']['medical']
    print(f"  Use for: {medical_guidelines['use_for']}")
    print(f"  Avoid for: {medical_guidelines['avoid_for']}")
    print(f"  Requirements: {medical_guidelines['requirements']}")
    print()
    
    print("Risk Mitigation:")
    risk_mitigation = recommendations['risk_mitigation']
    print(f"  Always Required: {risk_mitigation['always_required']}")
    print(f"  Critical Context: {risk_mitigation['context_specific']['critical']}")