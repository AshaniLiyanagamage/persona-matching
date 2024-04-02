import lime
import lime.lime_text
from .core import extract_sentence_embedding

explainer = lime.lime_text.LimeTextExplainer()

def explaination(companyPersona,candidatePersona):

    companyEducation = explainer.explain_instance(companyPersona['education'], extract_sentence_embedding, num_features=10)
    candidateEducation = explainer.explain_instance(candidatePersona['education'], extract_sentence_embedding, num_features=10)
    
    # companyTechSkills = explainer.explain_instance(companyPersona['technical_skills'], extract_sentence_embedding, num_features=10)
    # candidateTechSkills = explainer.explain_instance(candidatePersona['technical_skills'], extract_sentence_embedding, num_features=10)
    
    # companySoftSkills = explainer.explain_instance(companyPersona['technical_skills'], extract_sentence_embedding, num_features=10)
    # candidateSoftSkills = explainer.explain_instance(candidatePersona['technical_skills'], extract_sentence_embedding, num_features=10)
    
    # companyExperience = explainer.explain_instance(companyPersona['experience'], extract_sentence_embedding, num_features=10)
    # candidateExperience = explainer.explain_instance(candidatePersona['experience'], extract_sentence_embedding, num_features=10)
    
    results={
        "education":{
            "company":companyEducation.as_list(),
            "candidate":candidateEducation.as_list()
        },
        # "technicalSkills":{
        #     "company":companyTechSkills.as_list(),
        #     "candidate":candidateTechSkills.as_list()
        # },
        # "softSkills":{
        #     "company":companySoftSkills.as_list(),
        #     "candidate":candidateSoftSkills.as_list()
        # },
        # "experience":{
        #     "company":companyExperience.as_list(),
        #     "candidate":candidateExperience.as_list()
        # },
    }
    
    return results