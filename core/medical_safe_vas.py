import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import statistics

class MedicalSafeVAS:
    """
    Medical-Safe Value/Affective System (VAS) 
    
    This version separates clinical decision-making from experiential learning
    to maintain medical accuracy while still enabling system improvement.
    """
    
    def __init__(self):
        """Initialize Medical-Safe VAS"""
        self.clinical_decisions = []  # Separate clinical decision tracking
        self.learning_experiences = []  # Non-clinical learning only
        self.clinical_accuracy_metrics = {
            'correct_diagnoses': 0,
            'total_diagnoses': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        # Separate mindsets for different contexts
        self.clinical_mindset = {
            'evidence_weight': 1.0,        # Always prioritize evidence
            'guideline_adherence': 1.0,    # Always follow guidelines
            'safety_first': 1.0,           # Safety is paramount
            'uncertainty_handling': 0.9    # High caution for uncertainty
        }
        
        self.learning_mindset = {
            'experience_integration': 0.7,
            'pattern_recognition': 0.8,
            'continuous_improvement': 0.9,
            'feedback_responsiveness': 0.8
        }
    
    def process_clinical_case(self, 
                            case_data: Dict,
                            clinical_evidence: List[str],
                            guidelines: List[str],
                            decision_made: str,
                            outcome: Optional[str] = None) -> Dict:
        """
        Process clinical case with strict evidence-based approach
        
        Args:
            case_data: Patient case information
            clinical_evidence: Evidence supporting decision
            guidelines: Relevant clinical guidelines
            decision_made: The clinical decision made
            outcome: Actual outcome (for learning)
            
        Returns:
            Clinical decision record
        """
        
        clinical_record = {
            'timestamp': datetime.now().isoformat(),
            'case_id': case_data.get('case_id', 'unknown'),
            'case_type': case_data.get('type', 'general'),
            'evidence_quality': self._assess_evidence_quality(clinical_evidence),
            'guideline_compliance': self._check_guideline_compliance(guidelines, decision_made),
            'decision': decision_made,
            'confidence_level': self._calculate_clinical_confidence(clinical_evidence, guidelines),
            'outcome': outcome,
            'learning_flag': False  # Initially no learning until outcome confirmed
        }
        
        self.clinical_decisions.append(clinical_record)
        
        # Update accuracy metrics if outcome is available
        if outcome:
            self._update_accuracy_metrics(clinical_record)
        
        return clinical_record
    
    def _assess_evidence_quality(self, evidence: List[str]) -> float:
        """
        Assess quality of clinical evidence (simplified)
        In real implementation, this would use established evidence hierarchies
        """
        if not evidence:
            return 0.0
        
        quality_score = 0.0
        evidence_weights = {
            'systematic_review': 1.0,
            'rct': 0.9,
            'cohort_study': 0.7,
            'case_control': 0.6,
            'case_series': 0.4,
            'expert_opinion': 0.2
        }
        
        for item in evidence:
            for evidence_type, weight in evidence_weights.items():
                if evidence_type in item.lower():
                    quality_score += weight
                    break
            else:
                quality_score += 0.3  # Default for unclassified evidence
        
        return min(quality_score / len(evidence), 1.0)
    
    def _check_guideline_compliance(self, guidelines: List[str], decision: str) -> float:
        """
        Check compliance with clinical guidelines
        """
        if not guidelines:
            return 0.5  # Neutral when no guidelines available
        
        # Simplified compliance check
        # In real implementation, this would use structured guideline matching
        compliance_score = 0.8  # Default high compliance assumption
        
        return compliance_score
    
    def _calculate_clinical_confidence(self, evidence: List[str], guidelines: List[str]) -> float:
        """
        Calculate confidence in clinical decision based on evidence and guidelines
        """
        evidence_quality = self._assess_evidence_quality(evidence)
        guideline_support = self._check_guideline_compliance(guidelines, "")
        
        # Conservative confidence calculation
        confidence = (evidence_quality * 0.7 + guideline_support * 0.3)
        
        # Reduce confidence if evidence is limited
        if len(evidence) < 2:
            confidence *= 0.8
        
        return confidence
    
    def _update_accuracy_metrics(self, record: Dict):
        """
        Update accuracy metrics when outcome is known
        """
        self.clinical_accuracy_metrics['total_diagnoses'] += 1
        
        # Simplified outcome assessment
        # In real implementation, this would use structured outcome evaluation
        if record['outcome'] and 'correct' in record['outcome'].lower():
            self.clinical_accuracy_metrics['correct_diagnoses'] += 1
        elif record['outcome'] and 'false_positive' in record['outcome'].lower():
            self.clinical_accuracy_metrics['false_positives'] += 1
        elif record['outcome'] and 'missed' in record['outcome'].lower():
            self.clinical_accuracy_metrics['false_negatives'] += 1
    
    def process_learning_experience(self,
                                  experience_type: str,
                                  description: str,
                                  learning_value: float,
                                  safety_impact: float = 0.0) -> Dict:
        """
        Process non-clinical learning experiences safely
        
        Args:
            experience_type: Type of learning (training, feedback, etc.)
            description: Description of experience
            learning_value: Value of learning (0-1)
            safety_impact: Impact on patient safety (0-1, where 1 is high impact)
        """
        
        learning_record = {
            'timestamp': datetime.now().isoformat(),
            'type': experience_type,
            'description': description,
            'learning_value': learning_value,
            'safety_impact': safety_impact,
            'applicable_contexts': self._determine_learning_contexts(experience_type)
        }
        
        self.learning_experiences.append(learning_record)
        
        # Only update learning mindset, never clinical mindset
        if safety_impact < 0.3:  # Only low-risk learning affects mindset
            self._update_learning_mindset(learning_record)
        
        return learning_record
    
    def _determine_learning_contexts(self, experience_type: str) -> List[str]:
        """
        Determine which contexts this learning applies to
        """
        context_mapping = {
            'communication_training': ['patient_interaction', 'team_communication'],
            'diagnostic_feedback': ['diagnosis_process'],
            'procedure_training': ['clinical_procedures'],
            'research_update': ['evidence_evaluation'],
            'system_improvement': ['workflow_optimization']
        }
        
        return context_mapping.get(experience_type, ['general_learning'])
    
    def _update_learning_mindset(self, learning_record: Dict):
        """
        Safely update learning mindset without affecting clinical decisions
        """
        learning_value = learning_record['learning_value']
        
        # Conservative updates to learning mindset only
        if learning_value > 0.7:
            self.learning_mindset['continuous_improvement'] = min(
                self.learning_mindset['continuous_improvement'] * 1.02, 1.0
            )
    
    def get_clinical_decision_support(self, 
                                   case_context: Dict,
                                   available_evidence: List[str],
                                   relevant_guidelines: List[str]) -> Dict:
        """
        Provide decision support based strictly on evidence and guidelines
        NOT on past experiences or affective factors
        """
        
        support = {
            'evidence_assessment': self._assess_evidence_quality(available_evidence),
            'guideline_compliance': self._check_guideline_compliance(relevant_guidelines, ""),
            'confidence_factors': {
                'evidence_strength': len(available_evidence),
                'guideline_clarity': len(relevant_guidelines),
                'case_complexity': case_context.get('complexity', 'standard')
            },
            'recommendations': self._generate_clinical_recommendations(
                available_evidence, relevant_guidelines, case_context
            ),
            'safety_alerts': self._generate_safety_alerts(case_context),
            'learning_opportunities': self._identify_learning_opportunities(case_context)
        }
        
        return support
    
    def _generate_clinical_recommendations(self, 
                                         evidence: List[str], 
                                         guidelines: List[str],
                                         context: Dict) -> List[str]:
        """
        Generate evidence-based clinical recommendations
        """
        recommendations = []
        
        if not evidence:
            recommendations.append("⚠️ LIMITED EVIDENCE: Seek additional clinical evidence before proceeding")
        
        if not guidelines:
            recommendations.append("📋 GUIDELINES: Consult relevant clinical guidelines")
        
        if context.get('complexity') == 'high':
            recommendations.append("🔍 COMPLEX CASE: Consider specialist consultation")
        
        # Always prioritize safety
        recommendations.append("🛡️ SAFETY FIRST: Verify all contraindications and drug interactions")
        
        return recommendations
    
    def _generate_safety_alerts(self, context: Dict) -> List[str]:
        """
        Generate safety alerts based on case context
        """
        alerts = []
        
        if context.get('patient_age', 0) > 65:
            alerts.append("🧓 ELDERLY PATIENT: Consider age-adjusted dosing and interactions")
        
        if context.get('allergies'):
            alerts.append("🚨 ALLERGIES: Verify no contraindicated medications")
        
        if context.get('comorbidities'):
            alerts.append("⚕️ COMORBIDITIES: Check for condition interactions")
        
        return alerts
    
    def _identify_learning_opportunities(self, context: Dict) -> List[str]:
        """
        Identify learning opportunities without affecting clinical decisions
        """
        opportunities = []
        
        if context.get('rare_condition'):
            opportunities.append("📚 RARE CONDITION: Document case for educational purposes")
        
        if context.get('novel_treatment'):
            opportunities.append("🔬 NOVEL TREATMENT: Monitor outcomes for evidence generation")
        
        return opportunities
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics focused on clinical accuracy
        """
        total = self.clinical_accuracy_metrics['total_diagnoses']
        if total == 0:
            accuracy = 0.0
        else:
            accuracy = self.clinical_accuracy_metrics['correct_diagnoses'] / total
        
        return {
            'clinical_accuracy': accuracy,
            'total_cases': total,
            'false_positive_rate': self.clinical_accuracy_metrics['false_positives'] / max(total, 1),
            'false_negative_rate': self.clinical_accuracy_metrics['false_negatives'] / max(total, 1),
            'learning_experiences': len(self.learning_experiences),
            'system_maturity': min(total / 100, 1.0)  # Maturity based on case volume
        }
    
    def generate_safety_report(self) -> Dict:
        """
        Generate comprehensive safety report
        """
        metrics = self.get_performance_metrics()
        
        safety_status = "SAFE"
        if metrics['false_positive_rate'] > 0.1 or metrics['false_negative_rate'] > 0.05:
            safety_status = "CAUTION"
        if metrics['clinical_accuracy'] < 0.8:
            safety_status = "REVIEW_REQUIRED"
        
        return {
            'safety_status': safety_status,
            'performance_metrics': metrics,
            'clinical_mindset': self.clinical_mindset,
            'recommendations': self._generate_safety_recommendations(metrics),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_safety_recommendations(self, metrics: Dict) -> List[str]:
        """
        Generate safety recommendations based on performance
        """
        recommendations = []
        
        if metrics['clinical_accuracy'] < 0.9:
            recommendations.append("📈 ACCURACY: Increase focus on evidence-based decision making")
        
        if metrics['false_negative_rate'] > 0.03:
            recommendations.append("🔍 SENSITIVITY: Review screening protocols to reduce missed cases")
        
        if metrics['total_cases'] < 50:
            recommendations.append("📊 EXPERIENCE: System requires more cases for reliable performance assessment")
        
        recommendations.append("✅ CONTINUOUS MONITORING: Maintain regular performance reviews")
        
        return recommendations

# Example usage for medical context
if __name__ == "__main__":
    # Create medical-safe VAS
    medical_vas = MedicalSafeVAS()
    
    # Process a clinical case
    case_data = {
        'case_id': 'CASE001',
        'type': 'diagnostic',
        'patient_age': 45,
        'complexity': 'standard'
    }
    
    clinical_evidence = [
        'cohort_study: 85% sensitivity for diagnostic criteria',
        'clinical_guideline: Standard diagnostic protocol'
    ]
    
    guidelines = [
        'AMA Guidelines for Diagnosis',
        'Local Hospital Protocol'
    ]
    
    # Process clinical decision
    clinical_record = medical_vas.process_clinical_case(
        case_data=case_data,
        clinical_evidence=clinical_evidence,
        guidelines=guidelines,
        decision_made="Recommended diagnostic test X",
        outcome="correct diagnosis confirmed"
    )
    
    print("Clinical Decision Record:")
    print(f"Evidence Quality: {clinical_record['evidence_quality']:.2f}")
    print(f"Confidence Level: {clinical_record['confidence_level']:.2f}")
    print()
    
    # Get decision support
    support = medical_vas.get_clinical_decision_support(
        case_context=case_data,
        available_evidence=clinical_evidence,
        relevant_guidelines=guidelines
    )
    
    print("Clinical Decision Support:")
    for rec in support['recommendations']:
        print(f"  {rec}")
    print()
    
    # Process learning experience separately
    learning_record = medical_vas.process_learning_experience(
        experience_type="communication_training",
        description="Completed patient communication workshop",
        learning_value=0.8,
        safety_impact=0.1
    )
    
    # Generate safety report
    safety_report = medical_vas.generate_safety_report()
    print("Safety Report:")
    print(f"Status: {safety_report['safety_status']}")
    print(f"Clinical Accuracy: {safety_report['performance_metrics']['clinical_accuracy']:.2f}")
    print("Safety Recommendations:")
    for rec in safety_report['recommendations']:
        print(f"  {rec}")