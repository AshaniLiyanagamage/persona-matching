from transformers import ElectraTokenizer, ElectraModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


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



def extract_sentence_embedding(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs[0]
#         print(hidden_states.shape)
        if last_hidden_states is None:
            raise ValueError("Hidden states not found in model output.")
        sentence_embedding = torch.mean(last_hidden_states, dim=1)  # Consider the last layer's hidden states
    return sentence_embedding.cpu().detach().numpy()