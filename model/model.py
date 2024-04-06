
import numpy as np
from .core import get_sentence_embeddings,get_similarity

__version__ = "0.1.0"

 # Load pre-trained Electra tokenizer and model


def algorithm(companyPersona,candidateArray,w_technical_skills,w_education,w_soft_skills,w_experience):
    educationVector=[]
    technicalSkillsVector=[]
    softSkillsVector=[]
    experienceVector=[]
    educationVector.append(companyPersona['education'])
    technicalSkillsVector.append(companyPersona['technical_skills'])
    softSkillsVector.append(companyPersona['soft_skills'])
    experienceVector.append(companyPersona['experience'])

    for i in range(0,len(candidateArray)): 
        educationVector.append(candidateArray[i]['education'])
        technicalSkillsVector.append(candidateArray[i]['technical_skills'])
        softSkillsVector.append(candidateArray[i]['soft_skills'])
        experienceVector.append(candidateArray[i]['experience'])
    
    education=get_sentence_embeddings(educationVector)
    technicalSkills=get_sentence_embeddings(technicalSkillsVector)
    softSkills=get_sentence_embeddings(softSkillsVector)
    experience=get_sentence_embeddings(experienceVector)

    technicalSkillSet = get_similarity(technicalSkills)
    educationSet= get_similarity(education)
    softSkillSet= get_similarity(softSkills)
    minimumExperienceSet = get_similarity(experience)

    results=[]
    for i in range(0,len(candidateArray)):
        key="overall_candidate_"+str(i+1)
        educationKey="education_"+key
        technicalKey="technical_skills_"+key
        softSkillsKey="soft_skills_"+key
        experienceKey="experience_"+key
        educationValue=educationSet[i]*1 
        technicalValue=technicalSkillSet[i]*1 
        softSkillsValue=softSkillSet[i]*1
        experienceValue=minimumExperienceSet[i]*1# type: ignore
        
        value=(w_technical_skills*technicalSkillSet[i]+w_education*educationSet[i]+w_soft_skills*softSkillSet[i]+w_experience*minimumExperienceSet[i])*100
        results.append({
            key: value,
            educationKey:educationValue,
            technicalKey:technicalValue,
            softSkillsKey:softSkillsValue,
            experienceKey:experienceValue
        })

    return results