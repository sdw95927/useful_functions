from types import MethodType 

cytof_img.get_seg = MethodType(get_seg, cytof_img)
