import copy
import time

import numpy as np

import torch
from tqdm import tqdm
from sklearn import metrics
import utils
import gc
import os

class Coach:
    def __init__(self, trainset, devset, testset, model, opt, sched,args, logger):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.scheduler = sched
        self.args = args
        self.experiment = logger

        self.pretrained_model = None
        self.n_max_utterances_pretrain = self.trainset.n_max_utterances
        self.n_max_utterances = self.trainset.n_max_utterances

        self.dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
            "mosei": {"Negative": 0, "Positive": 1},
            "meld": {'neu': 0, 'sur': 1, 'fea': 2, 'sad': 3, 'joy': 4, 'dis': 5, 'ang': 6}

        }

        if args.emotion == "mosei_7":
            self.label_to_idx = {
                "Strong Negative": 0,
                "Weak Negative": 1,
                "Negative": 2,
                "Neutral": 3,
                "Positive": 4,
                "Weak Positive": 5,
                "Strong Positive": 6,
            }
        else:
            self.label_to_idx = self.dataset_label_dict[args.dataset]

        self.best_dev_f1 = None
        self.best_dev_acc = None
        self.best_epoch = None
        self.best_state = None

    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)
        print("Loaded corect.....")

    def train(self):
        self.experiment.debug(self.model)
        # Early stopping.
        best_dev_f1, best_epoch, best_state = (
            self.best_dev_f1,
            self.best_epoch,
            self.best_state,
        )
        best_dev_acc = self.best_dev_acc
        dev_accs = []
        dev_f1s = []
        test_f1s = []
        test_accs = []
        train_losses = []
        dev_losses = []
        test_losses = []
        best_test_f1 = None
        btf1 = 0.
        bta = 0.
        total_train_time = 0.
        total_dev_time = 0.
        total_test_time = 0.
        # Train

        best_epoch = 0
        best_acc_from_f1 = 0.
        for epoch in range(1, self.args.epochs + 1):
            train_loss, train_time = self.train_epoch(epoch)
            total_train_time += train_time

            self.trainset.set_batch()
            dev_f1, dev_loss, dev_acc,_,_,_, _, dev_time= self.evaluate()
            total_dev_time += dev_time

            if self.scheduler is not None and self.args.scheduler != "cosineLR_linearWarmUp" and self.args.scheduler != "cosineLR_linearWarmUp2":
                self.scheduler.step()

            test_f1, test_loss, test_acc,results, graphs, fused_emb, init_emb, test_time= self.evaluate(test=True)
            total_test_time += test_time

            current_lr = self.opt.get_lr()
            print(f"Epoch {epoch + 1}/{self.args.epochs}, LR: {current_lr:.8f}, Loss: {test_loss:.4f}")


            if test_f1 > btf1:
                torch.save(self.model,
                           f"./{self.args.save_model_checkpoint}/{self.args.modalities}_best_model.pt")
                np.save(os.path.join(os.getcwd(), self.args.save_analysis_path,
                                     'train_' + self.args.modalities + "_golds_preds.npy"), results)
                np.save(os.path.join(os.getcwd(), self.args.save_analysis_path,
                                     'train_' + self.args.modalities + "_graphs.npy"), graphs)
                np.save(os.path.join(os.getcwd(), self.args.save_analysis_path,
                                     'train_' + self.args.modalities + "_fused_emb.npy"), fused_emb)
                np.save(os.path.join(os.getcwd(), self.args.save_analysis_path,
                                     'train_' + self.args.modalities + "_init_emb.npy"), init_emb)
                self.experiment.info("Gold standard, Predictions, atv_graphs are saved as numpy array.")
                best_epoch = epoch
                btf1 = test_f1
                best_acc_from_f1 = test_acc
            self.experiment.info(f"[best_test_f1]:{btf1}, [best_test_acc]:{best_acc_from_f1}, [epoch]:{best_epoch}")

            bta = test_acc if test_acc > bta else bta

            self.experiment.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
            if best_dev_f1 is None or dev_f1 > best_dev_f1:

                best_dev_f1 = dev_f1
                best_state = self.model.state_dict()


            self.experiment.info("[Dev set] [acc {:.4f}]".format(dev_acc))
            if best_dev_acc is None or dev_acc > best_dev_acc:
                best_dev_acc = dev_acc

                self.experiment.info("Save the best emotion_heart model.")
            self.experiment.info("[Test set] [f1 {:.4f}]".format(test_f1))
            self.experiment.info("[Test set] [acc {:.4f}]".format(test_acc))

            dev_f1s.append(dev_f1)
            dev_accs.append(dev_acc)
            test_f1s.append(test_f1)
            test_accs.append(test_acc)
            train_losses.append(train_loss)
            dev_losses.append(dev_loss)
            test_losses.append(test_loss)

            save_loss_plot_path = os.path.join(os.getcwd(), self.args.save_analysis_path,
                                               'train_' + self.args.modalities + "_loss_plot.png")

            utils.plot_and_save_loss(train_losses, dev_losses, test_losses, filename=save_loss_plot_path)

            # if self.args.experiment_in_comet:
            #     self.experiment.log_metric("F1 Score (Dev)", dev_f1, epoch=epoch)
            #     self.experiment.log_metric("ACC Score (Dev)", dev_acc, epoch=epoch)
            #     self.experiment.log_metric("F1 Score (Test)", test_f1, epoch=epoch)
            #     self.experiment.log_metric("train_loss", train_loss, epoch=epoch)
            #     self.experiment.log_metric("val_loss", dev_loss, epoch=epoch)

        # The best

        self.model.load_state_dict(best_state)
        self.experiment.info("")
        self.experiment.info("Best in epoch {}:".format(best_epoch))
        dev_f1, _, dev_acc,_,_,_, _, _= self.evaluate()
        self.experiment.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
        test_f1, _, test_acc,results, _,_, _, _ = self.evaluate(test=True)
        self.experiment.info("[Test set] f1 {}".format(test_f1))
        self.experiment.info(f"\n['Hid Test f1 {btf1} \n acc {bta}]")
        if self.args.log_in_comet:
            self.experiment.log_metric("best_dev_f1", best_dev_f1, epoch=epoch)
            self.experiment.log_metric("best_test_f1", best_test_f1, epoch=epoch)

        print(f"train_time: {total_train_time} / dev_time: {total_dev_time} / test_time: {total_test_time}")
        print(f"Average - train_time: {total_train_time/self.args.epochs} / dev_time: {total_dev_time/self.args.epochs} / test_time: {total_test_time}/sef.args.epochs")
    
        return best_dev_f1, best_dev_acc, best_epoch, best_state, train_losses, dev_losses, dev_f1s, test_f1s, dev_accs, test_accs, test_losses

    def train_epoch(self, epoch):
        epoch_loss = 0
        epoch_acc = 0
        train_time = 0.
        self.model.train()
        self.opt.zero_grad()
        num_train_batches = len(self.trainset)
        for idx in tqdm(range(num_train_batches), desc="train epoch {}".format(epoch)):
            # self.model.zero_grad()
            data = copy.deepcopy(self.trainset[idx])

            for k, v in data.items():
                if data[k] is not None:
                    data[k] = v.to(self.args.device)

            torch.cuda.synchronize()
            start_time = time.time()

            loss, logits, labels ,_, _, _= self.model(data, self.n_max_utterances, train=True)
            epoch_loss += loss.item()

            loss.backward()
            self.opt.step()

            torch.cuda.synchronize()
            end_time = time.time()
            train_time += end_time - start_time

            self.opt.zero_grad()

            if self.args.scheduler == "cosineLR_linearWarmUp" or self.args.scheduler == "cosineLR_linearWarmUp2":
                self.scheduler.step()

            torch.cuda.empty_cache()
            del loss

            golds = labels.detach().cpu().numpy()
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            acc = metrics.accuracy_score(golds, preds)

            epoch_acc += acc

            gc.collect()

        epoch_loss /= num_train_batches
        epoch_acc = epoch_acc * 100 / num_train_batches
        self.experiment.info(
            "[Epoch %d] [Loss: %f] [Acc: %f] [Time: %f]"
            % (epoch, epoch_loss, epoch_acc, train_time)
        )
        return epoch_loss, train_time

    def evaluate(self, test=False):
        dev_loss = 0
        eval_time = 0
        if test:
            dataset = self.testset
            n_max_utterances = self.testset.n_max_utterances
        else:
            dataset = self.devset
            n_max_utterances = self.devset.n_max_utterances

        a_all, t_all, v_all = [], [], []
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            data = []
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = copy.deepcopy(dataset[idx])
                for k, v in data.items():
                    if data[k] is not None :
                        data[k] = v.to(self.args.device)

                torch.cuda.synchronize()
                start_time = time.time()
                if not test:
                    loss, logits, labels,_, _, _= self.model(data, n_max_utterances)
                    torch.cuda.synchronize()
                    end_time = time.time()
                else:
                    loss, logits, labels,embeddings, fused_emb, init_emb = self.model(data, n_max_utterances)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    init_emb = init_emb.permute(2,0,1).contiguous()
                    modals = embeddings.permute(2,0,1).contiguous()

                eval_time += end_time - start_time
                golds.append(labels.detach().cpu())
                preds.append(logits.detach().cpu())
                dev_loss += loss.item()
            golds = torch.cat(golds, dim=-1).numpy()
            preds = np.argmax(torch.cat(preds, dim=-1).numpy(), axis=1)
            results = np.stack([golds,preds], axis=0)

            graphs = None
            fused_emb_numpy = None
            init_emb_numpy = None
            if test:
                graphs = modals.detach().cpu().numpy()
                fused_emb_numpy = logits.detach().cpu().numpy()
                init_emb_numpy = init_emb.detach().cpu().numpy()


            f1 = metrics.f1_score(golds, preds, average="weighted")
            acc = metrics.accuracy_score(golds, preds)

            if test:
                print(
                    metrics.classification_report(
                        golds, preds, target_names=self.label_to_idx.keys(), digits=4, zero_division=0
                    )
                )

                if self.args.log_in_comet:
                    self.experiment.log_confusion_matrix(
                        golds,
                        preds,
                        labels=list(self.label_to_idx.keys()),
                        overwrite=True,
                    )

        return f1, dev_loss, acc, results, graphs, fused_emb_numpy, init_emb_numpy, eval_time

    def unimodal_inference(self):

        self.experiment.debug(self.model)
        # Early stopping.
        best_dev_f1, best_epoch, best_state = (
            self.best_dev_f1,
            self.best_epoch,
            self.best_state,
        )
        best_dev_acc = self.best_dev_acc
        dev_accs = []
        dev_f1s = []
        test_f1s = []
        test_accs = []
        train_losses = []
        dev_losses = []
        test_losses = []
        best_test_f1 = None
        btf1 = 0.
        bta = 0.
        total_train_time = 0.
        total_dev_time = 0.
        total_test_time = 0.
        # Train

        best_epoch = 0
        best_acc_from_f1 = 0.
        for epoch in range(1, self.args.epochs + 1):
            train_loss, train_time = self.train_epoch(epoch)
            total_train_time += train_time

            self.trainset.set_batch()
            dev_f1, dev_loss, dev_acc, _, _, _, _, dev_time = self.evaluate()
            total_dev_time += dev_time

            if self.scheduler is not None and self.args.scheduler != "cosineLR_linearWarmUp" and self.args.scheduler != "cosineLR_linearWarmUp2":
                self.scheduler.step()

            test_f1, test_loss, test_acc, results, graphs, fused_emb, init_emb, test_time = self.evaluate(test=True)
            total_test_time += test_time

            current_lr = self.opt.get_lr()
            print(f"Epoch {epoch + 1}/{self.args.epochs}, LR: {current_lr:.8f}, Loss: {test_loss:.4f}")

            if test_f1 > btf1:
                torch.save(self.model,
                           f"./{self.args.save_model_checkpoint}/{self.args.modalities}_best_model.pt")
                np.save(os.path.join(os.getcwd(), self.args.save_analysis_path,
                                     'train_' + self.args.modalities + "_golds_preds.npy"), results)
                np.save(os.path.join(os.getcwd(), self.args.save_analysis_path,
                                     'train_' + self.args.modalities + "_graphs.npy"), graphs)
                np.save(os.path.join(os.getcwd(), self.args.save_analysis_path,
                                     'train_' + self.args.modalities + "_fused_emb.npy"), fused_emb)
                np.save(os.path.join(os.getcwd(), self.args.save_analysis_path,
                                     'train_' + self.args.modalities + "_init_emb.npy"), init_emb)
                self.experiment.info("Gold standard, Predictions, atv_graphs are saved as numpy array.")
                best_epoch = epoch
                btf1 = test_f1
                best_acc_from_f1 = test_acc
            self.experiment.info(f"[best_test_f1]:{btf1}, [best_test_acc]:{best_acc_from_f1}, [epoch]:{best_epoch}")

            bta = test_acc if test_acc > bta else bta

            self.experiment.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
            if best_dev_f1 is None or dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_state = self.model.state_dict()

            self.experiment.info("[Dev set] [acc {:.4f}]".format(dev_acc))
            if best_dev_acc is None or dev_acc > best_dev_acc:
                best_dev_acc = dev_acc

                self.experiment.info("Save the best emotion_heart model.")
            self.experiment.info("[Test set] [f1 {:.4f}]".format(test_f1))
            self.experiment.info("[Test set] [acc {:.4f}]".format(test_acc))

            dev_f1s.append(dev_f1)
            dev_accs.append(dev_acc)
            test_f1s.append(test_f1)
            test_accs.append(test_acc)
            train_losses.append(train_loss)
            dev_losses.append(dev_loss)
            test_losses.append(test_loss)

            save_loss_plot_path = os.path.join(os.getcwd(), self.args.save_analysis_path,
                                               'train_' + self.args.modalities + "_loss_plot.png")

            utils.plot_and_save_loss(train_losses, dev_losses, test_losses, filename=save_loss_plot_path)

            # if self.args.experiment_in_comet:
            #     self.experiment.log_metric("F1 Score (Dev)", dev_f1, epoch=epoch)
            #     self.experiment.log_metric("ACC Score (Dev)", dev_acc, epoch=epoch)
            #     self.experiment.log_metric("F1 Score (Test)", test_f1, epoch=epoch)
            #     self.experiment.log_metric("train_loss", train_loss, epoch=epoch)
            #     self.experiment.log_metric("val_loss", dev_loss, epoch=epoch)

        # The best

        self.model.load_state_dict(best_state)
        self.experiment.info("")
        self.experiment.info("Best in epoch {}:".format(best_epoch))
        dev_f1, _, dev_acc, _, _, _, _ = self.evaluate()
        self.experiment.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
        test_f1, _, test_acc, results, _, _, _ = self.evaluate(test=True)
        self.experiment.info("[Test set] f1 {}".format(test_f1))
        self.experiment.info(f"\n['Hid Test f1 {btf1} \n acc {bta}]")
        if self.args.log_in_comet:
            self.experiment.log_metric("best_dev_f1", best_dev_f1, epoch=epoch)
            self.experiment.log_metric("best_test_f1", best_test_f1, epoch=epoch)

        print(f"train_time: {total_train_time} / dev_time: {total_dev_time} / test_time: {total_test_time}")
        print(
            f"Average - train_time: {total_train_time / self.args.epochs} / dev_time: {total_dev_time / self.args.epochs} / test_time: {total_test_time}/sef.args.epochs")

        return best_dev_f1, best_dev_acc, best_epoch, best_state, train_losses, dev_losses, dev_f1s, test_f1s, dev_accs, test_accs, test_losses
