from data import get_MELD_loaders
import argparse
    
import os
import torch

import models

import utils
import graphdata as gdt

from datetime import datetime as dt


log = utils.get_logger('./log/train.log')

def main(args1, args2):
    utils.set_seed(args1.seed)

    # load data
    log.info("Load finetuning dataset... Name: " + args1.dataset)
    data_dir = os.path.join(os.getcwd(), args1.data_dir_path, args1.dataset)
    args1.data = os.path.join(data_dir, "data_" + args1.dataset + ".pkl")

    data = utils.load_pkl(args1.data)
    # print(f"train set size: {len(data['train'])},  Dev set size: {len(data['dev'])}, Test set size: {len(data['test'])}")
    graph_trainset_file = os.path.join(data_dir, f"graph_trainset.pkl")
    graph_devset_file = os.path.join(data_dir, "graph_devset.pkl")
    graph_testset_file = os.path.join(data_dir, "graph_testset.pkl")
    graph_allset_file = os.path.join(data_dir, "graph_allset.pkl")

    if args1.dataset == "iemocap_4" or args1.dataset == "iemocap":

        if not os.path.exists(graph_trainset_file):
            trainset = gdt.iemocap_4_graphDataset(data["train"], 'train', args1)
            utils.save_pkl(trainset, graph_trainset_file)
        trainset = utils.load_pkl(graph_trainset_file)

        args1.n_max_utterances = trainset.n_max_utterances
        args1.n_max_speakers = trainset.n_max_speakers

        if not os.path.exists(graph_devset_file):
            devset = gdt.iemocap_4_graphDataset(data["dev"], 'dev', args1)
            utils.save_pkl(devset, graph_devset_file)
        devset = utils.load_pkl(graph_devset_file)

        if not os.path.exists(graph_testset_file):
            testset = gdt.iemocap_4_graphDataset(data["test"], 'test', args1)
            utils.save_pkl(testset, graph_testset_file)
        testset = utils.load_pkl(graph_testset_file)

    elif args1.dataset == "meld":
        train_loader, dev_loader, test_loader, all_loader = get_MELD_loaders(args1.batch_size, args1.data)

        if not os.path.exists(graph_trainset_file):
            trainset = gdt.meld_graphDataset(train_loader, 'train', args1)
            utils.save_pkl(trainset, graph_trainset_file)
        trainset = utils.load_pkl(graph_trainset_file)

        args1.n_max_utterances = trainset.n_max_utterances
        args1.n_max_speakers = trainset.n_max_speakers

        if not os.path.exists(graph_devset_file):
            devset = gdt.meld_graphDataset(dev_loader, 'dev', args1)
            utils.save_pkl(devset, graph_devset_file)
        devset = utils.load_pkl(graph_devset_file)

        if not os.path.exists(graph_testset_file):
            testset = gdt.meld_graphDataset(test_loader, 'test', args1)
            utils.save_pkl(testset, graph_testset_file)
        testset = utils.load_pkl(graph_testset_file)

        # if not os.path.exists(graph_allset_file):
        #     allset = gdt.meld_graphDataset(all_loader, 'all', args)
        #     utils.save_pkl(allset, graph_allset_file)
        # allset = utils.load_pkl(graph_allset_file)


    elif args1.dataset == "mosei":
        pass

    else:
        pass
    
    log.debug("Building emotionheart...")

    if args1.unimodal_inference and args1.modalities in ["a", "t", "v"]:
        model = torch.load(
            f"./{args1.save_model_checkpoint}/atv_best_model.pt",
            weights_only=False
        )

        print("--- Checking Model Parameters ---")
        # if args1.specific:
        #     for name, param in model.named_parameters():
        #         if not name.startswith("linear_fusion") and not name.startswith("classifier"):
        #             param.requires_grad=False
        #         print(f"'{name}'-{param.requires_grad}")
        #     print("--- Check Complete ---")
        # else:
        #     for name, param in model.named_parameters():
        #         if not name.startswith("encoder.") and not name.startswith("linear_fusion") and not name.startswith("classifier"):
        #             param.requires_grad=False
        #         print(f"'{name}'-{param.requires_grad}")
        #     print("--- Check Complete ---")
        for name, param in model.named_parameters():
            if not name.startswith("encoder.") and not name.startswith("linear_fusion") and not name.startswith("classifier"):
                param.requires_grad=False
            print(f"'{name}'-{param.requires_grad}")
        print("--- Check Complete ---")
        
        model.args = args1
        model.modalities = args1.modalities
        model.n_modalities = len(args1.modalities)
        model.encoder.args = args1

        model.encoder.n_modalities = len(args1.modalities)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters:     {total_params:,}")
    else:
        if args1.dataset =="iemocap":
            n_nodes = trainset.n_max_utterances
        else:
            n_nodes = max([trainset.n_max_utterances, testset.n_max_utterances, testset.n_max_utterances])
        encoder = models.EmotionHeartEncoder(args1, n_nodes)
        decoder = models.EmotionHeartDecoder(args1)
        model = models.EmotionHeartModel(args1, encoder, decoder).to(args1.device)

    opt1 = models.Optim(float(args1.learning_rate), int(args1.T), float(args1.max_grad_value), float(args1.weight_decay), int(args1.epochs), int(args1.n_train_dialogues // args1.batch_size))
    opt1.set_parameters(model.parameters(), args1.optimizer)
    sched1 = opt1.get_scheduler(args1.scheduler)

    coach = models.Coach(trainset, trainset, devset, testset, model, opt1, sched1, args1, args2, log)

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
    save_loss_plot_path = os.path.join(os.getcwd(), args1.save_analysis_path+'_'+args2.dataset,
                                       "loss_plot_"+ dt.now().strftime('%Y-%m-%d-%H-%M-%S')+".png")
    save_metrics_plot_path = os.path.join(os.getcwd(), args1.save_analysis_path+'_'+args2.dataset,
                                       "metrics_"+ dt.now().strftime('%Y-%m-%d-%H-%M-%S')+".png")
    utils.plot_and_save_loss(ret[4], ret[5], ret[10], filename=save_loss_plot_path)
    torch.save(metrics, save_metrics_plot_path)

if __name__ == "__main__":

    dataset1 = "iemocap"
    dataset2 = "iemocap"
    parser1 = argparse.ArgumentParser(description="pretraining_data")

    parser1.add_argument(
        "--dataset",
        type=str,
        # required=True,
        default="iemocap",
        choices=["iemocap", "iemocap_4", "mosei", "meld"],
        help="Dataset name."
    )
    # parser.add_argument(
    #     "--scheduler", type=str, default="reduceLR", help="Name of scheduler."
    # )

    # Modalities
    """ Modalities effects:
        -> dimentions of input vectors in dataset.py
        -> number of heads in transformer_conv in UnimodalEncoder.py"""
    # parser1.add_argument(
    #     "--modalities",
    #     type=str,
    #     default="atv",
    #     # required=True,
    #     choices=["a", "t", "v", "at", "tv", "av", "atv"],
    #     help="Modalities",
    # )
    
    parser1.add_argument(
        "--relation_type",
        default="eam",
        choices=["e", "ea", "eam"],
        help="Choose relation contruct type. e: interlocuter, a: intralocuter, m: intermodality",
    )

    parser1.add_argument(
        "--optimizer",
        default="adamw",
        choices=["sgd","rmsprop","adam","adamw"],
        help="Choose optimizer",
    )

    temp = parser1.parse_args()

    config_path = "config/"
    args1 = utils.get_config_args(parser1, config_path+temp.dataset+'_pretrain.yaml', dataset=temp.dataset)

    args1.num_edges = len(args1.relation_type)
    args1.num_degree = len(args1.relation_type)
    if args1.relation_type == "eam" and args1.modalities == "atv":
        args1.num_degree += 1

    args1.ffn_embed_dim = args1.encoder_embed_dim * args1.ffn_embed_scaler

    args1.dataset_embedding_dims = {
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

    # parser.add_argument(
    #     "--scheduler", type=str, default="reduceLR", help="Name of scheduler."
    # )
    parser2 = argparse.ArgumentParser(description="finetuning_data")
    parser2.add_argument(
        "--dataset",
        type=str,
        # required=True,
        default="iemocap",
        choices=["iemocap", "iemocap_4", "mosei", "meld"],
        help="Dataset name."
    )

    # Modalities
    """ Modalities effects:
        -> dimentions of input vectors in dataset.py
        -> number of heads in transformer_conv in UnimodalEncoder.py"""
    # parser2.add_argument(
    #     "--modalities",
    #     type=str,
    #     default="atv",
    #     # required=True,
    #     choices=["a", "t", "v", "at", "tv", "av", "atv"],
    #     help="Modalities",
    # )

    parser2.add_argument(
        "--relation_type",
        default="eam",
        choices=["e", "ea", "eam"],
        help="Choose relation contruct type. e: interlocuter, a: intralocuter, m: intermodality",
    )

    parser2.add_argument(
        "--optimizer",
        default="adam",
        choices=["sgd", "rmsprop", "adam", "adamw"],
        help="Choose optimizer",
    )

    temp2 = parser2.parse_args()

    args2 = utils.get_config_args(parser2, config_path+temp2.dataset+'.yaml', dataset=temp2.dataset)

    args2.num_edges = len(args2.relation_type)
    args2.num_degree = len(args2.relation_type)
    if args2.relation_type == "eam" and args2.modalities == "atv":
        args2.num_degree += 1

    args2.ffn_embed_dim = args2.encoder_embed_dim * args2.ffn_embed_scaler

    args2.dataset_embedding_dims = {
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
            "a": 300,
            "t": 600,
            "v": 342
        }
    }


    log.debug(args1)
    log.debug(args2)

    main(args1, args2)
