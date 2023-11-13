import pydicom
import pydicom._storage_sopclass_uids
from pydicom.dataset import Dataset, FileDataset
from PIL import Image
import skimage
 
name = "test"
 
# Populate required values for file meta information
meta = pydicom.Dataset()
meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
 
# build dataset
ds = Dataset()
ds.file_meta = meta
ds.fix_meta_info()
 
# unknown options
ds.is_little_endian = True
ds.is_implicit_VR = False
ds.SOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
ds.SeriesInstanceUID = pydicom.uid.generate_uid()
ds.StudyInstanceUID = pydicom.uid.generate_uid()
ds.FrameOfReferenceUID = pydicom.uid.generate_uid()
# ds.BitsStored = 16
# ds.BitsAllocated = 16
ds.SamplesPerPixel = 1
# ds.HighBit = 15
ds.ImagesInAcquisition = "1"
ds.InstanceNumber = 1
ds.ImagePositionPatient = r"0\0\1"
ds.ImageOrientationPatient = r"1\0\0\0\-1\0"
ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"
ds.RescaleIntercept = "0"
ds.RescaleSlope = "1"
# ds.PixelRepresentation = 1
 
# Case options
ds.PatientName = "Anonymous"
ds.PatientID = "123456"
ds.Modality = "MR"
ds.StudyDate = '20200225'
ds.ContentDate = '20200225'
 
def ensure_even(stream):
    # Very important for some viewers
    if len(stream) % 2:
        return stream + b"\x00"
    return stream
 
############################## Image data ##########################
pixel_data_list = [_ for _ in image_m_compress_2]

# required for pixel handler
ds.BitsStored = 8
ds.BitsAllocated = 8
ds.HighBit = 7
ds.PixelRepresentation = 0
'''
ds.BitsStored = 8
ds.BitsAllocated = 8
ds.HighBit = 7
ds.PixelRepresentation = 0
'''
 
# grayscale without compression
ds.PhotometricInterpretation = "MONOCHROME2"
ds.SamplesPerPixel = 1  # 1 color = 1 sampleperpixel
ds.file_meta.TransferSyntaxUID = pydicom.uid.JPEGBaseline8Bit  # '1.2.840.10008.1.2.4.50'
# ds.PixelData = np.array(pixel_data_list)
'''
bytesarray = bytearray()
for i in range(image_m_compress_2.shape[0]):
    bytesarray.extend(bytearray(cv2.imencode('.jpg', image_m_compress_2[i])[1].tostring()))
ds.PixelData = bytes(bytesarray)
'''
ds.PixelData = pydicom.encaps.encapsulate([cv2.imencode('.jpg', image_m_compress_2[i])[1].tostring() 
                                           for i in range(image_m_compress_2.shape[0])])
ds.NumberOfFrames = len(pixel_data_list)
 
# Image shape
ds['PixelData'].is_undefined_length = False
ds.Columns = pixel_data_list[0].shape[1]
ds.Rows = pixel_data_list[0].shape[0]
 
# validate and save
pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
new_filename = "/data/NFS_DeepLearningResults/SmallAnimalDataExchange/cnv/20230908172006_compress_v3.dicom"
ds.save_as(new_filename, write_like_original=False)
