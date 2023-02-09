########### Writing to a global list #############
from multiprocessing import Process, Manager

def manager_list_appending(L, i, train_set):
    L.append(train_set[i])
    
L = []
train_set = Dataset(train_indexes, info_df, augmentation=False, from_roi=True)
with Manager() as manager:
    L = manager.list()
    processes = []
    for i in tqdm(range(len(train_indexes))):
        for j in range(100):
            p = Process(target=manager_list_appending, args=(L, i, train_set))
            p.start()
            processes.append(p)
        
    for p in processes:
        p.join()
    
    L = list(L)
