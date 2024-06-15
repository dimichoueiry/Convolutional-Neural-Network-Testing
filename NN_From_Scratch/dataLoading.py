import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split

class ImportData:
    def __init__(self, data_path: str):
        """
        Initialize with the path to the data.
        
        :param data_path: Path to the data (CSV file or directory of images)
        """
        self.data_path = data_path
        self.data = None

    def load_csv_data(self) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        :return: Loaded data as a pandas DataFrame
        """
        if not self.data_path.endswith('.csv'):
            raise ValueError("Provided path is not a CSV file.")
        self.data = pd.read_csv(self.data_path)
        return self.data
    
    def load_img_data(self, img_size: tuple = (224, 224), grayscale: bool = False, normalize: list = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]) -> np.ndarray:
        """
        Load image data from a directory.
        
        :param img_size: Desired size for the images (width, height)
        :param grayscale: Whether to convert images to grayscale
        :return: Loaded images as a NumPy array
        """
        if not os.path.isdir(self.data_path):
            raise ValueError("The provided path is not a directory of images.")
        
        mean, std = normalize
        images = []
        for file_name in os.listdir(self.data_path):
            if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                img_path = os.path.join(self.data_path, file_name)
                img = Image.open(img_path)
                if grayscale:
                    img = img.convert('L')  # Convert to grayscale
                    mean = [mean[0]] # Convert mean to grayscale
                    std = [std[0]] # Convert std to grayscale
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0  # Normalize pixel values
                 # Normalize based on provided means and stds
                img_array = (img_array - mean) / std
                images.append(img_array)
        
        if not images:
            raise ValueError("No valid image files found in the directory.")
        
        self.data = np.array(images)
        return self.data
    
    def split_data(self, train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15):
        """
        Split the data into training, validation, and test sets.
        
        :param train_size: Proportion of data to use for training
        :param val_size: Proportion of data to use for validation
        :param test_size: Proportion of data to use for testing
        :return: Tuple of (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first.")
        
        if train_size + val_size + test_size != 1.0:
            raise ValueError("train_size, val_size, and test_size must sum to 1.0")
        
        if isinstance(self.data, pd.DataFrame):
            X = self.data.drop(columns=self.data.columns[-1]).values
            y = self.data[self.data.columns[-1]].values
        else:
            X = self.data
            y = None  # You can modify this if you have labels for image data
        
        # Split the data into train and temporary sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size)
        
        # Calculate new validation n size relative to the temporary dataset
        val_relative_size = val_size / (val_size + test_size)
        
        # Split temporary set into validation and test sets
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - val_relative_size))
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
