from data import get_MELD_loaders
import argparse

import os
import torch

import models

import utils
import graphdata as gdt

from datetime import datetime as dt


log = utils.get_logger('./log/train.log')

def main(args):
    utils.set_seed(args.seed)

    # load data
    log.info("Load finetuning dataset... Name: " + args.dataset)
    data_dir = os.path.join(os.getcwd(), args.data_dir_path, args.dataset)
    args.data = os.path.join(data_dir, "data_" + args.dataset + ".pkl")

    data = utils.load_pkl(args.data)
    # print(f"train set size: {len(data['train'])},  Dev set size: {len(data['dev'])}, Test set size: {len(data['test'])}")
    graph_trainset_file = os.path.join(data_dir, f"graph_trainset.pkl")
    graph_devset_file = os.path.join(data_dir, "graph_devset.pkl")
    graph_testset_file = os.path.join(data_dir, "graph_testset.pkl")
    graph_allset_file = os.path.join(data_dir, "graph_allset.pkl")

    if args.dataset == "iemocap_4" or args.dataset == "iemocap":

        if not os.path.exists(graph_trainset_file):
            trainset = gdt.iemocap_4_graphDataset(data["train"], 'train', args)
            utils.save_pkl(trainset, graph_trainset_file)
        trainset = utils.load_pkl(graph_trainset_file)

        args.n_max_utterances = trainset.n_max_utterances
        args.n_max_speakers = trainset.n_max_speakers

        if not os.path.exists(graph_devset_file):
            devset = gdt.iemocap_4_graphDataset(data["dev"], 'dev', args)
            utils.save_pkl(devset, graph_devset_file)
        devset = utils.load_pkl(graph_devset_file)

        if not os.path.exists(graph_testset_file):
            testset = gdt.iemocap_4_graphDataset(data["test"], 'test', args)
            utils.save_pkl(testset, graph_testset_file)
        testset = utils.load_pkl(graph_testset_file)


    elif args.dataset == "meld":
        train_loader, dev_loader, test_loader, all_loader = get_MELD_loaders(args.batch_size, args.data)

        if not os.path.exists(graph_trainset_file):
            trainset = gdt.meld_graphDataset(train_loader, 'train', args)
            utils.save_pkl(trainset, graph_trainset_file)
        trainset = utils.load_pkl(graph_trainset_file)

        args.n_max_utterances = trainset.n_max_utterances
        args.n_max_speakers = trainset.n_max_speakers

        # if not os.path.exists(graph_devset_file):
        #     devset = gdt.meld_graphDataset(dev_loader, 'dev', args)
        #     utils.save_pkl(devset, graph_devset_file)
        # devset = utils.load_pkl(graph_devset_file)

        if not os.path.exists(graph_testset_file):
            testset = gdt.meld_graphDataset(test_loader, 'test', args)
            utils.save_pkl(testset, graph_testset_file)
        testset = utils.load_pkl(graph_testset_file)

        if not os.path.exists(graph_allset_file):
            allset = gdt.meld_graphDataset(all_loader, 'all', args)
            utils.save_pkl(allset, graph_allset_file)
        allset = utils.load_pkl(graph_allset_file)



    elif args.dataset == "mosei":
        pass

    log.debug("Building graphormer...")

    if args.unimodal_inference and args.modalities in ["a", "t", "v"]:
        model = torch.load(
            f"./{args.save_model_checkpoint}/atv_best_model.pt",
            weights_only=False
        )

        print("--- Checking Model Parameters ---")
        if args.specific:
            for name, param in model.named_parameters():
                if not name.startswith("linear_fusion") and not name.startswith("classifier"):
                    param.requires_grad=False
                print(f"'{name}' {param.requires_grad}")
            print("--- Check Complete ---")
        else:
            for name, param in model.named_parameters():
                if not name.startswith("encoder.") and not name.startswith("linear_fusion") and not name.startswith("classifier"):
                    param.requires_grad=False
                print(f"'{name}' {param.requires_grad}")
            print("--- Check Complete ---")
        model.args = args
        model.modalities = args.modalities
        model.n_modalities = len(args.modalities)
        model.encoder.args = args

        model.encoder.n_modalities = len(args.modalities)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters:     {total_params:,}")
    else:
        if args.dataset =="iemocap":
            n_nodes = trainset.n_max_utterances
        else:
            n_nodes = max([trainset.n_max_utterances, testset.n_max_utterances, testset.n_max_utterances])
        encoder = models.EmotionHeartEncoder(args, n_nodes)
        model = models.EmotionHeartModel(args, encoder).to(args.device)


    opt = models.Optim(float(args.learning_rate), int(args.T), float(args.max_grad_value), float(args.weight_decay), int(args.epochs), int(args.n_train_dialogues // args.batch_size))
    opt.set_parameters(model.parameters(), args.optimizer)
    sched = opt.get_scheduler(args.scheduler)

    coach = models.Coach(trainset, testset, testset, model, opt, sched, args, log)

    # Train and eval
    log.info("Start training...")

    ret = coach.train()
    # Save.
    metrics = {
        "best_dev_f1": ret[0],
        "best_dev_acc": ret[1],
        "best_epoch": ret[2],
        "best_state": ret[3],
        "train_losses": ret[4],
        "dev_losses": ret[5],
        "dev_f1s": ret[6],
        "test_f1s": ret[7],
        "dev_accs": ret[8],
        "test_accs": ret[9],
        "test_losses": ret[10],
    }
    save_loss_plot_path = os.path.join(os.getcwd(), args.save_analysis_path,
                                       "loss_plot_"+ dt.now().strftime('%Y-%m-%d-%H-%M-%S')+".png")
    save_metrics_plot_path = os.path.join(os.getcwd(), args.save_analysis_path,
                                       "metrics_"+ dt.now().strftime('%Y-%m-%d-%H-%M-%S')+".png")
    utils.plot_and_save_loss(ret[4], ret[5], ret[10], filename=save_loss_plot_path)
    torch.save(metrics, save_metrics_plot_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="training_data")

    parser.add_argument(
        "--specific",
        type=str,
        default=True,
        choices=[True, False],
        help="whether to use a modality-specific model or not.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        # required=True,
        default="meld",
        choices=["iemocap", "meld"],
        help="Dataset name."
    )
    parser.add_argument(
        "--relation_type",
        default="eam",
        choices=["e", "ea", "eam"],
        help="Choose relation contruct type. e: interlocuter, a: intralocuter, m: intermodality",
    )

    parser.add_argument(
        "--optimizer",
        default="adamw",
        choices=["sgd","rmsprop","adam","adamw"],
        help="Choose optimizer",
    )

    temp = parser.parse_args()

    setting = "specific" if temp.specific else "agnostic"
    args = utils.get_config_args(parser, 'config/'+temp.dataset+'_'+setting+'.yaml', dataset=temp.dataset)

    args.num_edges = len(args.relation_type)
    args.num_degree = len(args.relation_type)
    if args.relation_type == "eam" and args.modalities == "atv":
        args.num_degree += 1

    args.ffn_embed_dim = args.encoder_embed_dim * args.ffn_embed_scaler

    args.dataset_embedding_dims = {
        "iemocap": {
            "a": 100,
            "t": 768,
            "v": 512,
        },
        "iemocap_4": {
            "a": 100,
            "t": 768,
            "v": 512,
        },
        "mosei": {
            "a": 80,
            "t": 768,
            "v": 35,
        },
        "meld": {
            "a":300,
            "t":600,
            "v":342
        }
    }

    main(args)
