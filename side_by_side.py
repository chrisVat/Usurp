import numpy as np
import random
random.seed(42)

forgetting_scores = np.load("forgetting/c10_forgetting_indices.npy")
embeddings = np.load("embeddings/cifar_10_trained/skmedoids_per_class_distances.npy")
# print(embeddings)
# print(embeddings.shape)
# exit()
# argsort embeddings
embeddings_og = np.argsort(embeddings)
embeddings = np.argsort(embeddings).tolist()
embeddings_reduced = embeddings - np.min(embeddings)
embeddings_reduced = embeddings_reduced / np.max(embeddings_reduced)
embeddings_reduced = embeddings_reduced.tolist()

first_nonzero = 23841 
percentage = 0.01
percentage = first_nonzero / len(embeddings)

selected_embeddings = []
total_looked_at = percentage * len(forgetting_scores)

related_forgetting_scores = forgetting_scores[:int(total_looked_at)]

for index in forgetting_scores:
    selected_embeddings.append(embeddings[index])
    if len(selected_embeddings) >= total_looked_at:
        break

related_percents = []
for index in forgetting_scores:
    related_percents.append(embeddings_reduced[index])
    if len(related_percents) >= total_looked_at:
        break


# get properties of selected embeddings

def show_list_stats(list):
    print("Mean:", np.mean(list))
    print("STD: ", np.std(list))
    print("Min:",np.min(list))
    print("Max", np.max(list))
    print("Median", np.median(list))
    print("25 Percentile: ", np.percentile(list, 25))
    print("75 Percentile: ", np.percentile(list, 75))

print("Forgetting stats: ")
show_list_stats(related_percents)

# graph selections on a line with a point where a selection was made
selected = []
def get_selected_uniform2():
    every_n = 1 / percentage
    selected = []
    for i in range(len(embeddings)):
        if i % every_n == 0:
            selected.append(embeddings[i])
        if len(selected) >= len(selected_embeddings):
            break
    while len(selected) < len(selected_embeddings):
        random_idx = random.randint(0, len(embeddings)-1)
        if random_idx not in selected:
            selected.append(embeddings[random_idx])
    selected = np.array(selected)
    return selected

def get_selected_uniform():
    every = (len(embeddings)-1) / total_looked_at
    selected = []
    for i in range(len(embeddings)):
        if int(i*every) not in selected and int(i*every) < len(embeddings): 
            selected.append(embeddings[int(i*every)])
        if len(selected) >= len(selected_embeddings):
            break
    while len(selected) < len(selected_embeddings):
        random_idx = random.randint(0, len(embeddings)-1)
        if random_idx not in selected:
            selected.append(embeddings[random_idx])
    selected = np.array(selected)
    return selected

def get_selected_uniform3():
    every = (len(embeddings)-1) / total_looked_at
    selected = []
    for i in range(len(embeddings)):
        if int(i*every) not in selected and int(i*every) < len(embeddings): 
            selected.append(embeddings[int(i*every)])
        if len(selected) >= len(selected_embeddings):
            break
    while len(selected) < len(selected_embeddings):
        random_idx = int(np.random.normal(0.5, 0.25) * len(embeddings))
        if random_idx not in selected and random_idx >= 0 and random_idx < len(embeddings):
            selected.append(embeddings[random_idx])
    selected = np.array(selected)
    return selected

def get_selected_random():
    selected = []
    while len(selected) < total_looked_at:
        # generate normal value iwht mean 
        random_idx = int(np.random.normal(np.mean(related_percents), np.std(related_percents))*len(embeddings))
        if random_idx not in selected and random_idx >= 0 and random_idx < len(embeddings):
            selected.append(embeddings[random_idx])
            print(len(selected))
    return selected


def get_selected_random_faster():
    selected = []
    while len(selected) < total_looked_at:
        # generate normal value iwht mean 
        random_idexes = np.random.normal(np.mean(related_percents), np.std(related_percents), 1000)*len(embeddings)
        for random_idx in random_idexes:
            random_idx = int(random_idx)
            if random_idx not in selected and random_idx >= 0 and random_idx < len(embeddings):
                selected.append(embeddings[random_idx])
                print(len(selected))
    return selected

selected = get_selected_uniform2()


overlap = 0
for embedding in selected:
    if embedding in related_forgetting_scores:
        overlap += 1
print("Overlap: ", overlap / len(selected))



"""
count = 0
for embedding in embeddings_og:
    if embedding in related_forgetting_scores:
        # print(count)
        count = 0
    else:
        count += 1
"""
        
# graph it
# import matplotlib.pyplot as plt
# plt.plot(related_forgetting_scores, [0 for _ in range(len(related_forgetting_scores))], 'ro')
# plt.show()


