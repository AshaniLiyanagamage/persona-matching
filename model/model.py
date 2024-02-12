import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.metrics.pairwise import cosine_similarity


__version__ = "0.1.0"

preprocess_url='https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
model_url='https://tfhub.dev/google/electra_base/2'
bert_preprocess_model=hub.KerasLayer(preprocess_url)
bert_encode_model = hub.KerasLayer(model_url)

def get_sentence_embeddings(sentences):
  text_preprocessed = bert_preprocess_model(sentences)
  return bert_encode_model(text_preprocessed)['pooled_output']

def get_similarity(vectors):
  count=vectors.get_shape()[0]
  results=[]
  for i in range (1,count):
    results.append(cosine_similarity([vectors[0]],[vectors[i]])[0][0])
  return results


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
    minimumExperienceSet = get_similarity(softSkills)

    results=[]
    for i in range(0,len(candidateArray)):
        results.append((w_technical_skills*technicalSkillSet[i]+w_education*educationSet[i]+w_soft_skills*softSkillSet[i]+w_experience*minimumExperienceSet[i])*100)

    return results