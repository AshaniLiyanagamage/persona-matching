import lime
import lime.lime_text
from .core import extract_sentence_embedding
import threading

explainer = lime.lime_text.LimeTextExplainer()

def explaination(companyPersona,candidatePersona,type):

    companyExplaination = explainer.explain_instance(companyPersona[type], extract_sentence_embedding, num_features=10)
    candidateExplaination = explainer.explain_instance(candidatePersona[type], extract_sentence_embedding, num_features=10)
    
    
    results={
        "response":{
            "company":companyExplaination.as_list(),
            "candidate":candidateExplaination.as_list()
        },
    }
    
    return results

