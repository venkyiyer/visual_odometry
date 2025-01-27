from PIL import Image
import numpy as np
import os

def load_pil_img(path_to_img, convert_to_rgb=False):
    '''
    This function load an img using the Image from PIL

    Parameters:
    - path_to_img: path of the img [str]
    - convert_to_rgb: Flag value [bool]

    Return:
    - PIL image
    '''
    if convert_to_rgb:
        return Image.open(path_to_img).convert("RGB")
    else:
        return Image.open(path_to_img)

def compute_relative_pose(SE3_1: np.ndarray, SE3_2: np.ndarray) -> np.ndarray:
    '''
    This function computes the relative pose given two poses in SE(3) representation

    Parameters:
    - SE3_1: prev_pose in SE3
    - SE3_2: curr_pose in SE3

    Returns:
    - relative pose between the two
    '''

    # Invert the first SE(3) matrix (SE3_1)
    # This is akin to changing the reference frame from the first pose to the world origin
    inverse_SE3_1 = np.linalg.inv(SE3_1)

    # Compute the relative pose by matrix multiplication
    # This operation essentially computes the transformation required to go
    # from the first pose (SE3_1) to the second pose (SE3_2)
    # The '@' symbol is used for matrix multiplication in Python
    SE3_relative = inverse_SE3_1 @ SE3_2

    # Return the computed relative pose in SE(3) representation
    return SE3_relative

def list_dir_with_relative_path_to(path, sort=True, mode='folder', extension=None, filter_by_word=None, revert_filter=False):
    '''
    Exctract the full path of a directory
    Parameters
    - path -> path to the folder [str]
    - sort -> True if we want to sort the result, False otherwise [bool]
    - mode -> can be 'folder' [default value] or 'file' [str]
    - extension -> None (if we want to extract all types of files) otherwise str (the extension))
    - revert_filter -> if true it will NOT extract the files that has contains the filter_word

    Return
    - list_of_abs_path -> list obj
    '''
    # inzializziamo le variabili
    list_of_abs_path = []

    # step 1: otteniamo il contenuto del path fornito
    root_content = os.listdir(path)

    if sort:
        root_content = sorted(root_content)

    # step 2: costruiamo il percorso assoluto
    for content in root_content:
        tmp = os.path.join(path, content)
        if mode == 'folder':
            # se stiamo raccogliendo solo cartella allora verifichiamo che esse siano cartelle
            if os.path.isdir(tmp):
                if filter_by_word is None:
                    list_of_abs_path.append(tmp)
                else:
                    if not revert_filter:
                        if filter_by_word in tmp:
                            list_of_abs_path.append(tmp)
                    elif revert_filter:
                        if filter_by_word not in tmp:
                            list_of_abs_path.append(tmp)

        elif mode == 'file' or mode == 'all':
            if extension is None:
                # non è stato definito alcun criterio di estrazione per estensione, quindi estraggo tutti i file
                if filter_by_word is None:
                    # non filtro
                    list_of_abs_path.append(tmp)
                else:
                    if not revert_filter:
                        if filter_by_word in tmp:
                            list_of_abs_path.append(tmp)
                    elif revert_filter:
                        if filter_by_word not in tmp:
                            list_of_abs_path.append(tmp)
            else:
                # se è stata definita una estensione da cercare
                if extension in tmp:
                    if filter_by_word is None:
                        # non filtro
                        list_of_abs_path.append(tmp)
                    else:
                        if not revert_filter:
                            if filter_by_word in tmp:
                                list_of_abs_path.append(tmp)
                        elif revert_filter:
                            if filter_by_word not in tmp:
                                list_of_abs_path.append(tmp)

    return list_of_abs_path

def load_InternalDT(path_to_dataset):
    '''
    Load the root directory of the UCBM internal dataset

    Parameters:
    - path_to_dataset: path to the dataset [str]

    Returns:
    - root_ucbm_content: the content of the root folder of the datasets [list[str]]
    '''

    root_ucbm_content = list_dir_with_relative_path_to(path_to_dataset)

    return root_ucbm_content


def load_EndoSlam(path_to_dataset, mode = "testing"):
    '''
    Load the root directory of the EndoSlam dataset. If mode is "training" then it will load the training part of the
    dataset, otherwise if mode is "testing" it will load the testing part.

    Parameters:
    - path_to_dataset: path to the dataset [str]
    - mode: can be training/testing [str]

    Returns:
    - root_EndoSLam_content: the content of the root folder of the datasets [list[str]]
    '''

    endoslam_content = list_dir_with_relative_path_to(path_to_dataset)
    return endoslam_content




class DatasetsIO:
    def list_dir_with_relative_path_to(self, path, sort=True, mode='folder', extension=None, filter_by_word=None, revert_filter=False):
        '''
        Exctract the full path of a directory
        Parameters
        - path -> path to the folder [str]
        - sort -> True if we want to sort the result, False otherwise [bool]
        - mode -> can be 'folder' [default value] or 'file' [str]
        - extension -> None (if we want to extract all types of files) otherwise str (the extension))
        - revert_filter -> if true it will NOT extract the files that has contains the filter_word

        Return
        - list_of_abs_path -> list obj
        '''
        # inzializziamo le variabili
        list_of_abs_path = []

        # step 1: otteniamo il contenuto del path fornito
        root_content = os.listdir(path)

        if sort:
            root_content = sorted(root_content)

        # step 2: costruiamo il percorso assoluto
        for content in root_content:
            tmp = os.path.join(path, content)
            if mode == 'folder':
                # se stiamo raccogliendo solo cartella allora verifichiamo che esse siano cartelle
                if os.path.isdir(tmp):
                    if filter_by_word is None:
                        list_of_abs_path.append(tmp)
                    else:
                        if not revert_filter:
                            if filter_by_word in tmp:
                                list_of_abs_path.append(tmp)
                        elif revert_filter:
                            if filter_by_word not in tmp:
                                list_of_abs_path.append(tmp)

            elif mode == 'file' or mode == 'all':
                if extension is None:
                    # non è stato definito alcun criterio di estrazione per estensione, quindi estraggo tutti i file
                    if filter_by_word is None:
                        # non filtro
                        list_of_abs_path.append(tmp)
                    else:
                        if not revert_filter:
                            if filter_by_word in tmp:
                                list_of_abs_path.append(tmp)
                        elif revert_filter:
                            if filter_by_word not in tmp:
                                list_of_abs_path.append(tmp)
                else:
                    # se è stata definita una estensione da cercare
                    if extension in tmp:
                        if filter_by_word is None:
                            # non filtro
                            list_of_abs_path.append(tmp)
                        else:
                            if not revert_filter:
                                if filter_by_word in tmp:
                                    list_of_abs_path.append(tmp)
                            elif revert_filter:
                                if filter_by_word not in tmp:
                                    list_of_abs_path.append(tmp)

        return list_of_abs_path
    def load_UCBM(self, path_to_dataset):
        '''
        Load the root directory of the UCBM internal dataset

        Parameters:
        - path_to_dataset: path to the dataset [str]

        Returns:
        - root_ucbm_content: the content of the root folder of the datasets [list[str]]
        '''

        root_ucbm_content = self.list_dir_with_relative_path_to(path_to_dataset)

        return root_ucbm_content

    def load_EndoSlam(self, path_to_dataset, mode = "testing"):
        '''
        Load the root directory of the EndoSlam dataset. If mode is "training" then it will load the training part of the
        dataset, otherwise if mode is "testing" it will load the testing part.

        Parameters:
        - path_to_dataset: path to the dataset [str]
        - mode: can be training/testing [str]

        Returns:
        - root_EndoSLam_content: the content of the root folder of the datasets [list[str]]
        '''

        endoslam_content = self.list_dir_with_relative_path_to(path_to_dataset)




        return endoslam_content

    def ucbm_dataloader(root_ucbm, batch_size, num_worker, i_folder):
        '''
        This function create the dataloader obj for the UCBM internal dataset

        Parameter:
        - root_ucbm: list of the root content [list[str]]
        - batch_size: the size of the batch to pass to the model during training
        - num_worker: the number of worker to use
        - i_folder: the ith folder to use in the list

        Return:
        - train_loader
        '''
        if i_folder >= len(root_ucbm):
            i_folder = 0

        # get a list of the content inside the ith folder
        rgb_folder = os.path.join(root_ucbm[i_folder], "rgb")
        depth_folder = os.path.join(root_ucbm[i_folder], "depth")
        training_frames = self.list_dir_with_relative_path_to(rgb_folder, mode = 'file', extension='.jpg', filter_by_word='dp', revert_filter=True)
        training_depth = self.list_dir_with_relative_path_to(depth_folder, mode='file', extension='.png')
        training_dataset = PoseDatasetLoader(training_frames, training_depth, dataset_type="UCBM")
        train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers = num_worker, pin_memory=True)

        return train_loader, i_folder


    def endoslam_dataloader(self, root_endoslam, batch_size, num_worker, i):
        '''
        This function create the dataloader obj for the EndoSlam dataset

        Parameter:
        - root_endoslam: list of the root content [list[str]]
        - batch_size: the size of the batch to pass to the model during training
        - num_worker: the number of worker to use
        - i_folder: the ith folder to use in the list

        Return:
        - test_loader
        '''
        rgb_folder = os.path.join(root_endoslam[i], "rgb")
        depth_folder = os.path.join(root_endoslam[i], "depth")
        testing_frames = self.list_dir_with_relative_path_to(rgb_folder, mode='file',
                                                        extension=".jpg", filter_by_word="dp", revert_filter=True)
        testing_depths = self.list_dir_with_relative_path_to(depth_folder, mode='file', extension='.png')

        testing_poses_path_xlsx = self.list_dir_with_relative_path_to(root_endoslam[i], mode="file", extension=".xlsx")
        print(testing_poses_path_xlsx)

        testing_poses = self.xlsxIO.read_xlsx_pose_file(testing_poses_path_xlsx[0])

        testing_dataset = PoseDatasetLoader(testing_frames, testing_depths, testing_poses, dataset_type="EndoSlam")
        test_loader = DataLoader(testing_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_worker,
                                 pin_memory=True)
        return test_loader