from transformers import ElectraTokenizer, ElectraModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


__version__ = "0.1.0"

 # Load pre-trained Electra tokenizer and model
tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
model = ElectraModel.from_pretrained('google/electra-base-discriminator')

def get_sentence_embeddings(sentences):
    results=[]
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs[0]
        sentence_embedding = torch.mean(last_hidden_states, dim=1)
        results.append(sentence_embedding)
    return results

def get_similarity(vectors):
    results = []
    # Convert PyTorch tensors to NumPy arrays and detach gradients if needed
    vectors_array = [tensor.detach().numpy() if tensor.requires_grad else tensor.numpy() for tensor in vectors]
    for i in range(1, len(vectors_array)):
        # Reshape the arrays to 2D with one row
        vector1 = vectors_array[0].reshape(1, -1)
        vector2 = vectors_array[i].reshape(1, -1)
        results.append(cosine_similarity(vector1, vector2)[0][0])
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