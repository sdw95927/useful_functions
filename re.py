    # Deep create folder
    sub_folder = re.search("(?<={}).*".format(data_dir), os.path.dirname(slide_file))[0]
    _output_dir = os.path.join(output_dir, *sub_folder.split("/"))  # sub_folder starting with '/' will overwrte output_dir
    if not os.path.exists(_output_dir):
        os.makedirs(_output_dir)
