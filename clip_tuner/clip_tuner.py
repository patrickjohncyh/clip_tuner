"""Main module."""
from comet_ml import Experiment
from torch import nn
from torch import optim
from torch_optimizer import AdaBelief
import clip
from clip.model import CLIP
import tqdm
import torch
from clip_tuner.dataset import ImageCaptioningDataset
from torch.utils.data import DataLoader
import gc
import numpy as np
import torch.distributed as dist
import wget


def convert_models_to_fp32(model):
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()


class CLIPDist(nn.Module):

    def __init__(self, model: CLIP):
        super(CLIPDist, self).__init__()
        self.model = model

    def forward(self, image, text):
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)
        # print('IN MODEL -- IMAGE FEAT SHAPE: {}'.format(image_features.shape))
        # print('IN MODEL -- TEXT FEAT SHAPE: {}'.format(text_features.shape))

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return image_features, self.model.logit_scale.exp() * text_features

class CLIPTuner:

    def __init__(self,
                 optimizer='adam',
                 lr=5e-5,
                 betas=(0.9, 0.98),
                 eps=1e-6,
                 weight_decay=0.2,
                 temperature=1.0,
                 comet_tracking=None,
                 multi_gpu:bool =False,
                 frozen_vision=True,
                 from_pretrained=None,
                 base_model="ViT-B/32",
                 **kwargs):

        assert optimizer in ['adam', 'adamw', 'adabelief']

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
        self.model, self.preprocess = clip.load(base_model, jit=False)  # Must set jit=False for training
        if from_pretrained is not None:
            print('DOWNLOADING WEIGHTS FROM {}'.format(from_pretrained))
            filename = wget.download(from_pretrained)
            print('FILENAME IS {}'.format(filename))
            print('USING ALTERNATE PRE-TRAINED WEIGHTS')
            self.model.load_state_dict(torch.load(filename))

        self.model = CLIPDist(self.model)

        if frozen_vision:
            for _ in self.model.model.visual.parameters():
                _.requires_grad = False


        self.model = nn.parallel.DataParallel(self.model.cuda()) if multi_gpu else self.model
        if comet_tracking:
            self.experiment = Experiment(comet_tracking)
        else:
            self.experiment = Experiment(project_name=kwargs.get('project_name',None))

        if self.device == "cpu":
            self.model.float()
        else:
            clip.model.convert_weights(self.model)

        hyper_params = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "temperature": temperature,
            "optimizer": optimizer
        }
        self.temperature = temperature

        self.experiment.log_parameters(hyper_params)

        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()








        if optimizer == 'adam':
            self.optimizer = optim.Adam([_ for _ in self.model.parameters() if _.requires_grad == True],
                                        lr=hyper_params["lr"],
                                        betas=hyper_params["betas"],
                                        eps=hyper_params["eps"],
                                        weight_decay=hyper_params["weight_decay"])
        elif optimizer == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(),
                                        lr=hyper_params["lr"],
                                        betas=hyper_params["betas"],
                                        eps=hyper_params["eps"],
                                        weight_decay=hyper_params["weight_decay"])
        elif optimizer == 'adabelief':
            self.optimizer = AdaBelief(self.model.parameters(),
                                        lr=hyper_params["lr"],
                                        betas=hyper_params["betas"],
                                        eps=hyper_params["eps"],
                                        weight_decay=hyper_params["weight_decay"])

    def forward_pass(self, list_image, list_txt, **kwargs):
        images = list_image
        # images = images.to(self.device)
        texts = clip.tokenize(list_txt, truncate=kwargs.get('truncate', False))#.to(self.device)
        image_features, text_features_scaled = self.model(images, texts)

        # print('IMAGE FEATURE SHAPE: {}'.format(image_features.shape))
        # print('TEXT FEATURE SHAPE: {}'.format(text_features_scaled.shape))

        logits_per_image = image_features @ text_features_scaled.t()
        logits_per_text = text_features_scaled @ image_features.t()

        # print('IMAGE LOGITS SHAPE: {}'.format(logits_per_image.shape))
        # print('TEXT LOGITS SHAPE: {}'.format(logits_per_text.shape))

        ground_truth = torch.arange(len(images), dtype=torch.long).cuda()# device=self.device)
        total_loss = (self.loss_img(logits_per_image, ground_truth) +
                      self.loss_txt(logits_per_text, ground_truth)) / 2
        return total_loss

    def tuner(self, train_dataframe, validation_dataframe, batch_size=4, epochs=5, evaluation_steps=500, **kwargs):
        train_dataset = ImageCaptioningDataset(train_dataframe, self.preprocess)
        validation_dataset = ImageCaptioningDataset(validation_dataframe, self.preprocess)
        train_dataloader = DataLoader(train_dataset, shuffle=True, drop_last=True, batch_size=batch_size)
        validation_dataloader = DataLoader(validation_dataset, drop_last=True, batch_size=batch_size)
        step = 0
        eval_step = 0
        state_dicts = {}

        with self.experiment.train():
            for epoch in range(epochs):
                pbar = tqdm.tqdm(position=0, total=len(train_dataloader))
                pbar.set_description(f"{epoch}/{epochs}")
                for batch in train_dataloader:
                    self.optimizer.zero_grad()

                    list_image, list_txt = batch
                    total_loss = self.forward_pass(list_image, list_txt, **kwargs)

                    self.experiment.log_metric("loss", total_loss.item(), step=step)
                    step = step + 1

                    total_loss.backward()
                    if self.device == "cpu":
                        self.optimizer.step()
                    else:
                        convert_models_to_fp32(self.model)
                        self.optimizer.step()
                        clip.model.convert_weights(self.model)
                    pbar.update(1)

                    val_loss = []
                    if step % evaluation_steps == 0:
                        for batch in validation_dataloader:
                            pbar.set_description("Currently Validating")
                            with torch.no_grad():
                                list_image, list_txt = batch
                                total_loss = self.forward_pass(list_image, list_txt, **kwargs)
                                val_loss.append(total_loss.item())

                        self.experiment.log_metric("validation_loss", np.mean(val_loss), step=step)

                        # store state_dict as cpu
                        state_dicts[eval_step] = {
                            'validation_loss': np.mean(val_loss),
                            'state_dict': { k.replace('module.model.',''): v.cpu() for k, v in self.model.state_dict().items()}
                        }

                        eval_step+=1
                pbar.close()

        # just keep the best 5 due to OOM issues
        steps_sorted_by_val_loss = sorted(state_dicts, key=lambda k: state_dicts[k]['validation_loss'])
        for k in steps_sorted_by_val_loss:
            if k not in steps_sorted_by_val_loss[:10]:
                del state_dicts[k]
                # ensure mem is freed
                gc.collect()



        return state_dicts
