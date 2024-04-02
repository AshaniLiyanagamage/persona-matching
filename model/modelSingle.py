import numpy as np
from .core import get_sentence_embeddings,get_similarity

def algorithm(companyPersona,candidatePersona,w_technical_skills,w_education,w_soft_skills,w_experience):
    educationVector=[]
    technicalSkillsVector=[]
    softSkillsVector=[]
    experienceVector=[]
    educationVector.append(companyPersona['education'])
    technicalSkillsVector.append(companyPersona['technical_skills'])
    softSkillsVector.append(companyPersona['soft_skills'])
    experienceVector.append(companyPersona['experience'])

    
    educationVector.append(candidatePersona['education'])
    technicalSkillsVector.append(candidatePersona['technical_skills'])
    softSkillsVector.append(candidatePersona['soft_skills'])
    experienceVector.append(candidatePersona['experience'])
    
    education=get_sentence_embeddings(educationVector)
    technicalSkills=get_sentence_embeddings(technicalSkillsVector)
    softSkills=get_sentence_embeddings(softSkillsVector)
    experience=get_sentence_embeddings(experienceVector)

    technicalSkillSet = get_similarity(technicalSkills)
    educationSet= get_similarity(education)
    softSkillSet= get_similarity(softSkills)
    minimumExperienceSet = get_similarity(experience)


    value=(w_technical_skills*technicalSkillSet[0]+w_education*educationSet[0]+w_soft_skills*softSkillSet[0]+w_experience*minimumExperienceSet[0])*100
    results={"response": value}

    return results