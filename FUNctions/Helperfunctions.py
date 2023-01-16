# -*- coding: utf-8 -*-
"""
Functions script

@author: Adam.Ruane
"""



def Unzipfiles(file_location, destination):
    """
      Add location of files to function to unzip  file and extract it in the working directory specified
      Parameters
      ----------
      file location : str
      where the files are stored
      -------------
      destination : str
      Where you will unzip files to.
      """
    for file in os.listdir(file_location):   
        if zipfile.is_zipfile(file): 
            with zipfile.ZipFile(file) as item:
                item.extractall(destination) 
                
   
def aggregatezippedfiles(working_directory, filetype):
    """
      Add location of zipped files and aggregate data to a new dataframe.
      Parameters
      ----------
      working directory : str
      where the files are stored
      -------------
      filetype : str
      type of file (usually ".xlsx")
      """
    files = os.listdir(working_directory)
    end = filetype
    df = pd.DataFrame()
    for filename in files:    
        if filename.endswith(end) in filename:
            Gdf = pd.read_excel(working_directory+filename)
            res = [df, Gdf]
            df = pd.concat(res, axis=0)
            
            
  