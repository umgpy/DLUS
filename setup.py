from setuptools import setup, find_namespace_packages

setup(name='DLUS',
      packages=find_namespace_packages(include=["DLUS"]),
      description='DLUS: Deep Learning-based Segmentation of Prostatic Urethra on Computed Tomography Scans for Treatment Planning',
      url='https://github.com/BSEL-UC3M/DLUS',
      author='IGT - BSEL-UC3M',
      author_email='jpascau@ing.uc3m.es',
      install_requires=[
            "dcmrtstruct2nii",
          "dicom2nifti",
          "itk",
          "keras",
          "Keras-Preprocessing",
          "matplotlib",
          "MedPy",
          "nibabel",
          "nnunet==1.7.1",
          "pandas",
          "pydicom",
          "pydicom-seg",
          "scikit-image",
          "scikit-learn",
          "scipy",
          "SimpleITK",
          "scikit-learn",
          "tensorflow==2.7.0",
          "torch==1.10.1",
          "tqdm"
      ],
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation', 'DLUS']
      )
