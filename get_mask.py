from scipy.stats import describe
from openslide import open_slide
from segmentation_functions import parseXML, createMask

slide_feature_df = pd.DataFrame()
index = 0
for patient_id in patient_paired:
    print("----------------- {} ----------------".format(patient_id))
    
    origin_feature_dfs = []
    for out_dir in meta_dict[patient_id]['PT']:
        origin_feature_df = pd.read_csv(os.path.join(out_dir, "cell_summary_20X.csv"))
        slide_id = out_dir.split("/")[-1][:-4]
        slide_file = os.path.join(slide_folder, meta_df.loc[slide_id, "Image path"], str(slide_id)+".svs")
        xml_file = os.path.join(slide_folder, meta_df.loc[slide_id, "Image path"], str(slide_id)+".xml")
        if os.path.exists(xml_file):
            #************ Only retain nuclei within ROI *************
            xml = parseXML(xml_file, "ROI")
            if len(xml['ROI']) > 0:
                slide = open_slide(slide_file)
                mask = createMask(slide.level_dimensions, xml, "ROI").T
                zoom = slide.level_dimensions[0][0] / slide.level_dimensions[-1][0]
                if slide.properties['aperio.AppMag'] == '20':
                    nuclei_xs = np.array(origin_feature_df.loc[:, "coordinate_x"].values/zoom/2, dtype="int")
                    nuclei_ys = np.array(origin_feature_df.loc[:, "coordinate_y"].values/zoom/2, dtype="int")
                elif slide.properties['aperio.AppMag'] == '40':
                    nuclei_xs = np.array(origin_feature_df.loc[:, "coordinate_x"].values/zoom, dtype="int")
                    nuclei_ys = np.array(origin_feature_df.loc[:, "coordinate_y"].values/zoom, dtype="int")
                nuclei_ys[nuclei_ys > mask.shape[0] - 1] = mask.shape[0] - 1
                nuclei_xs[nuclei_xs > mask.shape[1] - 1] = mask.shape[1] - 1
                origin_feature_df = origin_feature_df.loc[mask[nuclei_ys, nuclei_xs] == 1, :]
            #************ Exclude nuclei within excluded *************
            xml = parseXML(xml_file, "excluded")
            if len(xml['excluded']) > 0:
                slide = open_slide(slide_file)
                mask = createMask(slide.level_dimensions, xml, "excluded").T
                zoom = slide.level_dimensions[0][0] / slide.level_dimensions[-1][0]
                if slide.properties['aperio.AppMag'] == '20':
                    nuclei_xs = np.array(origin_feature_df.loc[:, "coordinate_x"].values/zoom/2, dtype="int")
                    nuclei_ys = np.array(origin_feature_df.loc[:, "coordinate_y"].values/zoom/2, dtype="int")
                elif slide.properties['aperio.AppMag'] == '40':
                    nuclei_xs = np.array(origin_feature_df.loc[:, "coordinate_x"].values/zoom, dtype="int")
                    nuclei_ys = np.array(origin_feature_df.loc[:, "coordinate_y"].values/zoom, dtype="int")
                nuclei_ys[nuclei_ys > mask.shape[0] - 1] = mask.shape[0] - 1
                nuclei_xs[nuclei_xs > mask.shape[1] - 1] = mask.shape[1] - 1
                origin_feature_df = origin_feature_df.loc[mask[nuclei_ys, nuclei_xs] == 0, :]
