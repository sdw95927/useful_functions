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

############### joblib ######################
from joblib import Parallel, delayed
def match_scale_parallel(self, scale_min=0.7, scale_max=1.3, scale_step=0.05, angle_min=-10, angle_max=10, angle_step=1, 
                         parallel_rotation=False, n_jobs=20):
    start = time.time()
    if parallel_rotation:
        match_rotation_method = self.match_rotation_parallel
    else:
        match_rotation_method = self.match_rotation
    
    def _match_rotation(self, scale):
        img_source_scaled = skimage.transform.rescale(self.img_source, scale, preserve_range=True)
        max_value_rotation, info_rotation = match_rotation_method(angle_min, angle_max, angle_step, 
                                                                img_source=img_source_scaled)
        max_value_rotation /= scale 
        return max_value_rotation, info_rotation + scale
    
    scales = np.arange(scale_min, scale_max, step=scale_step)
    res = Parallel(n_jobs=n_jobs)(delayed(_match_rotation)(self, scale) for scale in scales)
    max_value = -np.inf
    for _max_value, _info in res:
        if _max_value > max_value:
            max_value = _max_value
            info = _info
    
    print("Time used: {}s".format(time.time() - start))
    
    return max_value, info
