import os
import numpy as np
import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from PIL import Image


class Evaluation:
    def __init__(self):
        self.jaccard_score = 0.
        self.dice = 0.
        self.spec = 0.
        self.sens = 0.
        self.acc = 0.

    def __call__(self, gtdir, resdir, resprefix):

        listimgfiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(gtdir, 'val_*.bmp'))]
        for currFile in tqdm(listimgfiles):
            res = np.array(Image.open(os.path.join(resdir, currFile + '_' + resprefix + '.bmp')).convert('L'))
            res = np.float32(res)/255

            gt = np.array(Image.open(os.path.join(gtdir, currFile + '.bmp')).convert('L'))
            gt = np.float32(gt)/255

            self.jaccard_score += self.jaccard_similarity_coefficient(gt, res)
            self.dice += self.dice_coefficient(gt, res)
            spec_tmp, sens_tmp, acc_tmp = self.specificity_sensitivity(gt, res)
            self.spec += spec_tmp
            self.sens += sens_tmp
            self.acc += acc_tmp

            self.dice /= len(listimgfiles)
            self.jaccard_score /= len(listimgfiles)
            self.spec /= len(listimgfiles)
            self.sens /= len(listimgfiles)
            self.acc /= len(listimgfiles)

    def print_vals(self):
        print('DiceCoefficient: {}\n'
              'JaccardIndex: {}\n'
              'Specificity: {}\n'
              'Sensitivity: {}\n'
              'Accuracy: {}'.format(self.dice, self.jaccard_score, self.spec, self.sens, self.acc))

    def dice_coefficient(self, res, gt):  
        
        dice = np.zeros(res.shape[0])
        for i in range(res.shape[0]):
          A = gt[i,:,:].flatten()
          B = res[i,:,:,:].flatten()

          A = np.array([1 if x > 0.5 else 0.0 for x in A])
          B = np.array([1 if x > 0.5 else 0.0 for x in B])
          dice[i] = np.sum(A * B)*2.0 / (np.sum(B) + np.sum(A) + np.finfo(np.float32).eps)
          
        return dice.mean()

    def specificity_sensitivity(self, gt, res):

        specificity=np.zeros(res.shape[0])
        sensitivity=np.zeros(res.shape[0])
        accuracy=np.zeros(res.shape[0])
        for i in range(res.shape[0]):
            A = gt[i, :, :].flatten()
            B = res[i, :, :, :].flatten()

            A = np.array([1 if x > 0.5 else 0.0 for x in A])
            B = np.array([1 if x > 0.5 else 0.0 for x in B])

            tn, fp, fn, tp = np.float32(confusion_matrix(A, B, labels=[0, 1]).ravel())
            specificity[i] = tn/(fp + tn + np.finfo(np.float32).eps)
            sensitivity[i] = tp/(tp + fn + np.finfo(np.float32).eps)
            accuracy[i] = (tp + tn)/(tp + fp + fn + tn + np.finfo(np.float32).eps)
        # sensitivity=(tp)/(tp+fp+fn)
        # accuracy=(2*tp)/(2*tp+fp+fn)

        return specificity.mean(), sensitivity.mean(),accuracy.mean()

    def jaccard_similarity_coefficient(self, res, gt, no_positives=1.0):

        J = np.zeros(res.shape[0])

        for i in range(res.shape[0]):
            A = gt[i,:,:].flatten()
            B = res[i,:,:].flatten()

            A = np.array([1 if x > 0.5 else 0.0 for x in A])
            B = np.array([1 if x > 0.5 else 0.0 for x in B])

            intersect = np.minimum(A,B)
            union = np.maximum(A, B)

        # Special case if neither A or B have a 1 value.
            if union.sum() == 0:
               return no_positives

            J[i] = float(intersect.sum()) / union.sum()

        return J.mean()

    def cls(self, gt, res):

        A = 1 if gt > 0.5 else 0.0
        B = 1 if res > 0.5 else 0.0
        accu = 0
        if A == B:
            accu = 1

        return accu
