import matplotlib.pyplot as plt                   
import torch
import numpy as np
from datetime import datetime
from sklearn.manifold import TSNE
import sys
visualizer = sys.modules[__name__]

import seaborn as sns
import os

from src.utils.evaluator import center_normed_tensor2img
from src.utils.utils import createLogHandler

pylogger = createLogHandler(__name__, 'training_log.log')
DATASET_EXAMPLE_DIR = 'reports/dataset_example'
SEED = 666
NOW = str(datetime.now())

def save_dataset_example_as_png(cfg, dataset, sample_num=2, train_or_test='train', is_class_showed=False, figsize=(20, 10)):
    indices = np.random.choice(len(dataset), sample_num)
    if 'Single' in str(dataset.__class__):
        print_single_dataset_example(dataset, indices, is_class_showed, figsize=figsize)
    elif 'Triplet' in str(dataset.__class__):
        print_triplet_dataset_example(dataset, indices, is_class_showed, figsize=figsize)

    title = f'{train_or_test} dataset example {NOW}'
    plt.suptitle(title)        
    plt.tight_layout()
    save_path = os.path.join(cfg['project_path'], DATASET_EXAMPLE_DIR, 'dataset' + cfg['data']['dataset_postfix'], f'{title}.png')
    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))
    plt.savefig(save_path)

def print_single_dataset_example(dataset, indices, is_class_showed=False, figsize=(20, 10)):
    plt.figure(figsize=figsize)
    for i, index in enumerate(indices):
        tensor1, label = dataset[index]
        if len(tensor1.shape) == 4:
            tensor1 = tensor1[0]
        img1 = center_normed_tensor2img(tensor1)                                        
        plt.imshow(img1)            
         
def print_triplet_dataset_example(dataset, indices, is_class_showed=False, figsize=(20, 10)):
    fig, axes = plt.subplots(len(indices), 3, figsize=figsize)
    for i, index in enumerate(indices):
        (tensor1, tensor2, tensor3), label, _ = dataset[index]
        if len(tensor1.shape) == 4:
            tensor1 = tensor1[0]
            tensor2 = tensor2[0]
            tensor3 = tensor3[0]
        anc_label, pos_label, neg_label = dataset.get_label(index) if is_class_showed else (None, None, None)
        img1 = center_normed_tensor2img(tensor1)
        img2 = center_normed_tensor2img(tensor2)
        img3 = center_normed_tensor2img(tensor3)
        
        anchor_title = f'anchor:{anc_label}'
        positive_title = f'ground truth: positive:{pos_label}'                
        negative_title = f'ground truth: negative:{neg_label}'
                
        axes[i][0].imshow(img1)
        axes[i][0].set_title(anchor_title, fontsize=20)   

        axes[i][1].imshow(img2)
        axes[i][1].set_title(positive_title, fontsize=20)
                
        axes[i][2].imshow(img3)
        axes[i][2].set_title(negative_title, fontsize=20) 


class MetricVisualizer:
    def setup(self, figure_dir, roc_steps=400, task_seed=666):
        self.figure_dir = figure_dir
        self.roc_steps = roc_steps
        self.task_seed = task_seed 

    def plot_auc_roc(self, fprs, tprs, auc, data_type, model_status):
        plt.figure()
        lw = 2        
        pylogger.info(f'auc: {auc}')
        plt.plot(fprs, tprs, color='darkorange',
                lw=lw, label=f'ROC curve (area = {auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        title = f'ROC curve \n[auc:{auc:.4f}][seed:{self.task_seed}][data:{data_type}][model:{model_status}]'
        plt.title(title)
        plt.legend(loc="lower right")

        plt.savefig(os.path.join(self.figure_dir, title.replace("\n", "")+f'_{NOW}.png'))

    def plot_total_acc(self, thresholds, accuracies, data_type, model_status):
        plt.figure()
        plt.plot(thresholds, accuracies, color='darkorange', label=f'acc')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        best_idx = np.argmax(accuracies)
        best_acc = np.max(accuracies)
        best_thres = thresholds[best_idx]
        pylogger.info(f'best acc: {best_acc}, thres: {best_thres}')
        
        title = f'Total Accuracy \n[best:{best_acc:.4f}][threshold:{best_thres:.4f}][seed:{self.task_seed}][data:{data_type}][model:{model_status}]'
        plt.title(title)
        plt.savefig(os.path.join(self.figure_dir, title.replace("\n", "")+f'_{NOW}.png'))

    def plot_k_fold_best_acc(self, best_metrcis, data_type, model_status):
        k_fold_best_accuracies = best_metrcis['k_fold_best_accuracies']
        n_folds = k_fold_best_accuracies.size
        pylogger.info(f'mean k fold acc: {k_fold_best_accuracies.mean()}, {k_fold_best_accuracies}')
        plt.figure()
        plt.bar(range(len(k_fold_best_accuracies)), k_fold_best_accuracies)
        plt.ylim([0.0, 1.0])
        title = f'{n_folds}-Fold Best Accuracy [mean:{k_fold_best_accuracies.mean():.4f}][seed:{self.task_seed}][data:{data_type}][model:{model_status}]'
        plt.title(title)
        plt.savefig(os.path.join(self.figure_dir, title.replace("\n", "")+f'_{NOW}.png'))

    def plot_confusion_matrix_with_best_threshold(self, best_metrcis, data_type, model_status):
        plt.figure()
        import pandas as pd
        threshold = best_metrcis['threshold']
        TN, FP, FN, TP = best_metrcis['TN'], best_metrcis['FP'], best_metrcis['FN'], best_metrcis['TP']
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        f1_score = 2*(recall*precision) / (recall+precision)
        df = pd.DataFrame([[TN, FP], 
                           [FN, TP]], index=['Not the same', 'Same'], columns=['Not the same', 'Same'])

        sns.heatmap(df, annot=True, fmt='n')
        plt.xlabel('Prediction')
        plt.ylabel('Ground Truth')
        pylogger.info(f'f1_score: {f1_score}, precision: {precision}, recall: {recall}')
        title = f'Confution Matrix [thres:{threshold:.4f}][f1_score:{f1_score:.4f}]\n[precision:{precision:.4f}][recall:{recall:.4f}][seed:{self.task_seed}][data:{data_type}][model:{model_status}]'
        plt.title(title)
        plt.savefig(os.path.join(self.figure_dir, title.replace("\n", "")+f'_{NOW}.png'))


    def visualize(self, thresholds_thresholds_metrics_dict, best_metrcis, data_type='test', model_status='after training'):
        fprs = thresholds_thresholds_metrics_dict['FPRs']
        tprs = thresholds_thresholds_metrics_dict['TPRs']
        thresholds = thresholds_thresholds_metrics_dict['thresholds']
        accuracies = thresholds_thresholds_metrics_dict['accuracies']
        auc = thresholds_thresholds_metrics_dict['AUC']
        self.plot_auc_roc(fprs, tprs, auc, data_type, model_status)
        self.plot_total_acc(thresholds, accuracies, data_type, model_status)
        self.plot_k_fold_best_acc(best_metrcis, data_type, model_status)
        self.plot_confusion_matrix_with_best_threshold(best_metrcis, data_type, model_status)

def tsne(embeddings_multidim, n_components=2):
    from sklearn.manifold import TSNE
    return TSNE(n_components=n_components, learning_rate='auto', init='random', random_state=SEED).fit_transform(embeddings_multidim)    

def pca(embeddins_multi_dim, n_components=2):
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components, random_state=SEED).fit_transform(embeddins_multi_dim)

class EmbeddingVisualizer:
    def setup(self, 
        figure_dir, 
        sampled=True,
        sample_size_per_batch=10, 
        sampled_classes_num=10, 
        plot_in_2d_or_3d=2, 
        device=torch.device('cpu'),
        task_seed=666
    ):
        self.figure_dir = figure_dir
        self.sampled = sampled
        self.sample_size_per_batch = sample_size_per_batch
        self.plot_in_2d_or_3d = plot_in_2d_or_3d
        self.device = device
        self.embeddings = None
        self.labels = None
        self.transformed_embeddings = None
        np.random.seed(SEED)
        self.colors = [np.random.choice(256, 3)/255 for _ in range(sampled_classes_num)]
        self.sampled_classes_num = sampled_classes_num
        self.task_seed = task_seed 

    def extract_embeddings(self, dataloader, model):
        with torch.no_grad():
            model.eval()
            model.to(self.device)
            total_batches = len(dataloader)
            x, y = next(iter(dataloader))
            
            size_per_batch = self.sample_size_per_batch if self.sampled else len(y)

            self.embeddings = np.zeros((total_batches*size_per_batch, x.shape[1]))
            self.labels = np.zeros(total_batches*size_per_batch)
            k = 0
            for images, targets in dataloader:
                indices = np.random.choice(targets.shape[0], self.sample_size_per_batch) if self.sampled else np.arange(targets.shape[0])
                images = images[indices].to(self.device)
                targets = targets[indices].cpu().numpy()
                
                self.embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
                self.labels[k:k+len(images)] = targets
                k += len(images)

    def plot_embeddings(self, dim_reduction_method='pca', data_type='train', model_status='after training'):
        assert self.transformed_embeddings.shape[1] in [2, 3], 'Embeddings should be transformed to 2d or 3d'
        plt.figure(figsize=(10,10))
        plt.tight_layout()
        for i in range(self.sampled_classes_num):
            inds = np.where(self.labels==i)[0]
            plt.scatter(self.transformed_embeddings[inds, 0], self.transformed_embeddings[inds, 1], alpha=0.5, color=self.colors[i])

        plt.legend(np.unique(self.labels))
        title = f'Embedding space [{dim_reduction_method}][data:{data_type}][model:{model_status}][seed:{self.task_seed}])'
        plt.title(title)
        plt.savefig(os.path.join(self.figure_dir, f'{title}_{NOW}.png'))

    def visualize(self, dataloader, model, dim_reduction_method='pca', data_type='train', model_status='after training'):        
        self.embeddings = None
        self.labels = None
        self.transformed_embeddings = None

        self.extract_embeddings(dataloader, model)                
        if self.embeddings.shape[1] == self.plot_in_2d_or_3d:
            self.transformed_embeddings = self.embeddings

        elif dim_reduction_method == 'both':
            self.transformed_embeddings = pca(self.embeddings)
            self.plot_embeddings('pca', data_type, model_status)
            self.transformed_embeddings = tsne(self.embeddings)
            self.plot_embeddings('tsne', data_type, model_status)
            return 

        elif dim_reduction_method == 'pca':
            self.transformed_embeddings = pca(self.embeddings)

        elif dim_reduction_method == 'tsne':
            self.transformed_embeddings = tsne(self.embeddings)

        self.plot_embeddings(dim_reduction_method, data_type, model_status)


class DistanceVisualizer:
    def setup(self, figure_dir):
        self.figure_dir = figure_dir

    def visualize(
        self, 
        model_dis_dict,
        data_type='train', 
        figsize=(15, 5)
        ):
        """
        Args:
            model_dis_dict: nested dict having the following structure:
                    {
                        'model1_name': {'positive': positvie_distances_in_1d_array, 
                                    'negative': negatvie_distances_in_1d_array},
                        'model2_name': {...}
                    }        
        """
            
        model_names = list(model_dis_dict.keys())
        model_num = len(model_names)
        fig, axes = plt.subplots(1, model_num, figsize=figsize)
        for i in range(model_num):    
            model_name = model_names[i]
            output_distance = model_dis_dict[model_name]
            pos_mean = output_distance['positive'].mean()
            neg_mean = output_distance['negative'].mean()

            sns.distplot(output_distance['positive'], ax=axes[i])
            sns.distplot(output_distance['negative'], ax=axes[i])
        
            axes[i].legend(labels=[f'pos (mean: {pos_mean:.4f})', f'neg (mean: {neg_mean:.4f})'])
            axes[i].set_title(f'model_{"_".join(model_name.split(" "))} (data num: {output_distance["positive"].shape[0]})')
            axes[i].set_xlabel('distance')
            axes[i].axvline(pos_mean, color='navy', linestyle='--')
            axes[i].text(pos_mean, 0.05, f'{pos_mean:.4f}')
            axes[i].axvline(neg_mean, color='red', linestyle='-')
            axes[i].text(neg_mean, 0.05, f'{neg_mean:.4f}')
        data_num = output_distance['positive'].shape[0]
        title = f'Distribution of distance [data_num:{data_num}][data:{data_type}][model:{",".join(model_names)}])'
        plt.suptitle(title)
        plt.savefig(os.path.join(self.figure_dir, f'{title}_{NOW}.png'))


class PredictionVisualizer:        
    def setup(self, figure_dir, norm_method='center', sample_num=20, device=torch.device('cpu'), is_class_showed=False):
        self.figure_dir = figure_dir        
        self.norm_method = norm_method
        self.sample_num = sample_num
        self.device = device
        self.is_class_showed = is_class_showed

    def tensor2img(self, tensor):        
        return getattr(visualizer, self.norm_method + '_normed_tensor2img')(tensor)

    def visualize(self, dataset, model, threshold, pos_neg_distance_dict, indices_list=None, data_type='train', model_status='after training'):
        """[summary]

        Args:
            dataset ([type]): [description]
            model ([type]): [description]
            threshold ([type]): [description]
            pos_neg_distance_dict ([type]): [description]
            indices_list ([type], optional): [description]. Defaults to None. If passed then target sample will be selected to show.
            data_type (str, optional): [description]. Defaults to 'train'.
            model_status (str, optional): [description]. Defaults to 'after training'.
        """
        anc_label, pos_label, neg_label = '', '', ''
        model.to(self.device)
        
        pos_distance_arr = pos_neg_distance_dict['positive']
        neg_distance_arr = pos_neg_distance_dict['negative']        
        pos_sample_indices_in_test_dataset = pos_neg_distance_dict['pos_sample_indices_in_test_dataset']
        neg_sample_indices_in_test_dataset = pos_neg_distance_dict['neg_sample_indices_in_test_dataset']
        pos_sample_indices_in_test_dataset_sorted_asc = pos_sample_indices_in_test_dataset[np.argsort(pos_distance_arr)]
        neg_sample_indices_in_test_dataset_sorted_asc = neg_sample_indices_in_test_dataset[np.argsort(neg_distance_arr)]

        split_num = (self.sample_num-2)/2
        # show the most TP and TN first then show the most FP and FN
        if indices_list is None:
            indices_list = [pos_sample_indices_in_test_dataset_sorted_asc[0]] + \
                        [neg_sample_indices_in_test_dataset_sorted_asc[-1]] + \
                        list(pos_sample_indices_in_test_dataset_sorted_asc[-1*int(np.ceil(split_num)):]) + \
                        list(neg_sample_indices_in_test_dataset_sorted_asc[:int(np.floor(split_num))])
            
        fig, axes = plt.subplots(len(indices_list), 3, figsize=(20, 120))
        
        pylogger.info(f'Sample indices: {indices_list}')
        with torch.no_grad():
            for i, index in enumerate(indices_list):
                index = int(index)
                if self.is_class_showed:
                    anc_label, pos_label, neg_label = dataset.get_label(index)
                    
                (tensor1, tensor2, tensor3), target, _ = dataset[index]
                tensor1, tensor2, tensor3 = [tensor.to(self.device) for tensor in [tensor1, tensor2, tensor3]]
                anchor, positive, negative = model(tensor1.unsqueeze(0), tensor2.unsqueeze(0), tensor3.unsqueeze(0))
                anchor, positive, negative = anchor.cpu(), positive.cpu(), negative.cpu()
                pos_distance = pos_distance_arr[index]
                neg_distance = neg_distance_arr[index]
                img1 = self.tensor2img(tensor1)
                img2 = self.tensor2img(tensor2)
                img3 = self.tensor2img(tensor3)    

                anchor_title = f'anchor:{anc_label}'
                positive_title = f'ground truth: positive:{pos_label}'                
                negative_title = f'ground truth: negative:{neg_label}'
                
                axes[i][0].imshow(img1)
                axes[i][0].set_title(anchor_title, fontsize=20)   

                axes[i][1].imshow(img2)
                axes[i][1].set_title(positive_title, fontsize=20)
                if pos_distance < threshold:
                    pred = 'pos'  
                    color = 'b'
                else: 
                    pred = 'neg'  
                    color = 'r'
                axes[i][1].text(5, 5, f'distance: {pos_distance:.4f}, pred: {pred}', fontsize=15, c=color)
                
                axes[i][2].imshow(img3)
                axes[i][2].set_title(negative_title, fontsize=20) 
                if neg_distance < threshold:
                    pred = 'pos'  
                    color = 'r'
                else: 
                    pred = 'neg'  
                    color = 'b'
                axes[i][2].text(5, 5, f'distance: {neg_distance:.4f}, pred: {pred}', fontsize=15, c=color)                
        title = f'Prediction Result [model:{model_status}][data:{data_type}][threshold:{threshold:.4f}])'
        plt.suptitle(title, y=0.999, fontsize=20)
        plt.subplots_adjust(top=0.95)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, f'{title}_{NOW}.png'))