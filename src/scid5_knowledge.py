"""
SCID-5 (Structured Clinical Interview for DSM-5) 知识库
包含精神疾病诊断的结构化问诊流程
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class DisorderType(Enum):
    """精神疾病类型"""
    MOOD_DISORDERS = "mood_disorders"  # 心境障碍
    ANXIETY_DISORDERS = "anxiety_disorders"  # 焦虑障碍
    PSYCHOTIC_DISORDERS = "psychotic_disorders"  # 精神病性障碍
    SUBSTANCE_USE = "substance_use"  # 物质使用障碍
    PERSONALITY_DISORDERS = "personality_disorders"  # 人格障碍

@dataclass
class Question:
    """问诊问题数据结构"""
    id: str
    text: str
    category: DisorderType
    follow_up_conditions: Dict[str, str]  # 根据回答决定后续问题
    severity_assessment: bool = False  # 是否评估严重程度

@dataclass
class DiagnosticCriteria:
    """诊断标准"""
    disorder_name: str
    criteria: List[str]
    required_symptoms: int
    duration_requirement: str

class SCID5Knowledge:
    """SCID-5知识库"""
    
    def __init__(self):
        self.questions = self._initialize_questions()
        self.diagnostic_criteria = self._initialize_criteria()
        self.current_assessment_path = []
    
    def _initialize_questions(self) -> Dict[str, Question]:
        """初始化问诊问题库"""
        questions = {
            # 抑郁症筛查
            "depression_screening": Question(
                id="depression_screening",
                text="我想了解一下您最近的心情状况。您觉得自己最近的情绪怎么样？有什么特别的感受想要分享吗？",
                category=DisorderType.MOOD_DISORDERS,
                follow_up_conditions={
                    "yes": "depression_interest",
                    "no": "anxiety_screening"
                }
            ),
            
            "depression_interest": Question(
                id="depression_interest",
                text="在过去的两周内，您是否对平时感兴趣或愉快的活动明显失去兴趣？",
                category=DisorderType.MOOD_DISORDERS,
                follow_up_conditions={
                    "yes": "depression_symptoms",
                    "no": "depression_symptoms"
                }
            ),
            
            "depression_symptoms": Question(
                id="depression_symptoms",
                text="在过去的两周内，您是否经历过以下症状？请告诉我哪些适用：\n1. 食欲明显改变（增加或减少）\n2. 睡眠问题（失眠或睡眠过多）\n3. 精神运动性激越或迟滞\n4. 疲劳或精力不足\n5. 感到无价值或过度的罪恶感\n6. 注意力不集中或犹豫不决\n7. 反复想到死亡或自杀念头",
                category=DisorderType.MOOD_DISORDERS,
                follow_up_conditions={
                    "multiple": "depression_severity",
                    "few": "anxiety_screening"
                },
                severity_assessment=True
            ),
            
            "depression_severity": Question(
                id="depression_severity",
                text="这些症状对您的日常生活、工作或社交活动造成了多大的影响？（轻微/中等/严重）",
                category=DisorderType.MOOD_DISORDERS,
                follow_up_conditions={
                    "severe": "suicide_risk",
                    "moderate": "anxiety_screening",
                    "mild": "anxiety_screening"
                }
            ),
            
            "suicide_risk": Question(
                id="suicide_risk",
                text="您是否有过伤害自己或结束生命的想法？如果有，您是否制定过具体的计划？",
                category=DisorderType.MOOD_DISORDERS,
                follow_up_conditions={
                    "yes_plan": "emergency_referral",
                    "yes_no_plan": "safety_planning",
                    "no": "anxiety_screening"
                }
            ),
            
            # 焦虑症筛查
            "anxiety_screening": Question(
                id="anxiety_screening",
                text="在过去的6个月内，您是否经常感到过度的担心或焦虑，难以控制？",
                category=DisorderType.ANXIETY_DISORDERS,
                follow_up_conditions={
                    "yes": "anxiety_symptoms",
                    "no": "psychotic_screening"
                }
            ),
            
            "anxiety_symptoms": Question(
                id="anxiety_symptoms",
                text="当您感到焦虑时，是否伴随以下身体症状？\n1. 坐立不安或感到紧张\n2. 容易疲劳\n3. 注意力难以集中\n4. 易怒\n5. 肌肉紧张\n6. 睡眠障碍",
                category=DisorderType.ANXIETY_DISORDERS,
                follow_up_conditions={
                    "multiple": "anxiety_impact",
                    "few": "panic_screening"
                }
            ),
            
            "panic_screening": Question(
                id="panic_screening",
                text="您是否经历过突然的强烈恐惧或不适感，伴随心跳加速、出汗、颤抖等症状？这种感觉通常在几分钟内达到高峰？",
                category=DisorderType.ANXIETY_DISORDERS,
                follow_up_conditions={
                    "yes": "panic_frequency",
                    "no": "psychotic_screening"
                }
            ),
            
            # 精神病性症状筛查
            "psychotic_screening": Question(
                id="psychotic_screening",
                text="您是否曾经听到别人听不到的声音，或者看到别人看不到的东西？",
                category=DisorderType.PSYCHOTIC_DISORDERS,
                follow_up_conditions={
                    "yes": "psychotic_delusions",
                    "no": "substance_screening"
                }
            ),
            
            "psychotic_delusions": Question(
                id="psychotic_delusions",
                text="您是否曾经坚信一些别人认为不真实的事情，比如有人在监视您或者您有特殊的能力？",
                category=DisorderType.PSYCHOTIC_DISORDERS,
                follow_up_conditions={
                    "yes": "psychotic_impact",
                    "no": "substance_screening"
                }
            ),
            
            # 物质使用筛查
            "substance_screening": Question(
                id="substance_screening",
                text="在过去的12个月内，您的饮酒或药物使用是否对您的生活造成了问题？",
                category=DisorderType.SUBSTANCE_USE,
                follow_up_conditions={
                    "yes": "substance_details",
                    "no": "assessment_complete"
                }
            )
        }
        
        return questions
    
    def _initialize_criteria(self) -> Dict[str, DiagnosticCriteria]:
        """初始化诊断标准"""
        criteria = {
            "major_depression": DiagnosticCriteria(
                disorder_name="抑郁障碍",
                criteria=[
                    "情绪低落",
                    "兴趣丧失",
                    "食欲改变",
                    "睡眠障碍",
                    "精神运动改变",
                    "疲劳",
                    "无价值感/罪恶感",
                    "注意力问题",
                    "死亡念头"
                ],
                required_symptoms=5,
                duration_requirement="持续至少2周"
            ),
            
            "generalized_anxiety": DiagnosticCriteria(
                disorder_name="广泛性焦虑障碍",
                criteria=[
                    "过度焦虑和担心",
                    "难以控制担心",
                    "坐立不安",
                    "容易疲劳",
                    "注意力集中困难",
                    "易怒",
                    "肌肉紧张",
                    "睡眠障碍"
                ],
                required_symptoms=3,
                duration_requirement="持续至少6个月"
            ),
            
            "panic_disorder": DiagnosticCriteria(
                disorder_name="惊恐障碍",
                criteria=[
                    "反复的惊恐发作",
                    "对惊恐发作的持续担心",
                    "行为的显著改变"
                ],
                required_symptoms=2,
                duration_requirement="持续至少1个月"
            )
        }
        
        return criteria
    
    def get_initial_question(self) -> Question:
        """获取初始问题"""
        return self.questions["depression_screening"]
    
    def get_next_question(self, current_question_id: str, response: str) -> Optional[Question]:
        """根据当前问题和回答获取下一个问题"""
        print(f"🔍 DEBUG - GET_NEXT_QUESTION: current_question_id={current_question_id}, response={response}")
        
        current_question = self.questions.get(current_question_id)
        if not current_question:
            print(f"🔍 DEBUG - 未找到问题: {current_question_id}")
            return None
        
        # 记录评估路径
        self.current_assessment_path.append({
            "question_id": current_question_id,
            "response": response
        })
        
        # 改进的回答分类逻辑
        response_lower = response.lower()
        
        # 特殊处理自杀风险评估
        if current_question_id == "suicide_risk":
            if any(word in response_lower for word in ["计划", "具体", "方法"]):
                response_key = "yes_plan"
            elif any(word in response_lower for word in ["想法", "念头"]):
                response_key = "yes_no_plan"
            else:
                response_key = "no"
        else:
            # 一般问题的回答分类
            if any(word in response_lower for word in ["是", "有", "yes", "经常", "严重", "可以", "好", "持续", "周", "天", "月"]):
                response_key = "yes"
            elif any(word in response_lower for word in ["否", "没有", "no", "很少", "轻微", "不"]):
                response_key = "no"
            elif any(word in response_lower for word in ["多个", "几个", "多种", "很多"]):
                response_key = "multiple"
            else:
                response_key = "few"
        
        # 获取下一个问题ID
        next_question_id = current_question.follow_up_conditions.get(response_key)
        print(f"🔍 DEBUG - response_key={response_key}, initial_next_question_id={next_question_id}")
        print(f"🔍 DEBUG - follow_up_conditions={current_question.follow_up_conditions}")
        
        # 如果没有找到对应的后续问题，尝试使用默认路径
        if not next_question_id:
            print(f"🔍 DEBUG - 未找到对应的后续问题，尝试默认路径")
            # 为depression_screening等问题提供默认路径
            if current_question_id == "depression_screening":
                next_question_id = current_question.follow_up_conditions.get("yes")  # 默认继续抑郁症评估
                print(f"🔍 DEBUG - depression_screening默认路径: {next_question_id}")
            else:
                # 其他情况下，尝试获取第一个可用的后续问题
                available_keys = list(current_question.follow_up_conditions.keys())
                if available_keys:
                    next_question_id = current_question.follow_up_conditions[available_keys[0]]
                    print(f"🔍 DEBUG - 使用第一个可用路径: {next_question_id}")
        
        print(f"🔍 DEBUG - final_next_question_id={next_question_id}")
        
        if next_question_id == "assessment_complete":
            print(f"🔍 DEBUG - 评估完成")
            return None
        
        next_question = self.questions.get(next_question_id)
        print(f"🔍 DEBUG - next_question={next_question.id if next_question else None}")
        return next_question
    
    def generate_assessment_summary(self) -> str:
        """生成评估总结"""
        if not self.current_assessment_path:
            return "评估尚未开始"
        
        summary = "## 评估总结\n\n"
        
        # 分析症状模式
        mood_symptoms = []
        anxiety_symptoms = []
        psychotic_symptoms = []
        
        for item in self.current_assessment_path:
            question = self.questions.get(item["question_id"])
            if question and question.category == DisorderType.MOOD_DISORDERS:
                if any(word in item["response"].lower() for word in ["是", "有", "yes", "经常"]):
                    mood_symptoms.append(question.text)
            elif question and question.category == DisorderType.ANXIETY_DISORDERS:
                if any(word in item["response"].lower() for word in ["是", "有", "yes", "经常"]):
                    anxiety_symptoms.append(question.text)
            elif question and question.category == DisorderType.PSYCHOTIC_DISORDERS:
                if any(word in item["response"].lower() for word in ["是", "有", "yes"]):
                    psychotic_symptoms.append(question.text)
        
        # 生成建议
        if mood_symptoms:
            summary += "**心境相关症状:** 检测到抑郁相关症状，建议进一步评估\n"
        
        if anxiety_symptoms:
            summary += "**焦虑相关症状:** 检测到焦虑相关症状，建议进一步评估\n"
        
        if psychotic_symptoms:
            summary += "**精神病性症状:** 检测到可能的精神病性症状，强烈建议专业评估\n"
        
        summary += "\n**重要提醒:** 此评估仅供参考，不能替代专业医疗诊断。如有症状，请及时就医。"
        
        return summary

# 全局知识库实例
scid5_kb = SCID5Knowledge() 