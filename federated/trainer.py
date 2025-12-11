# federated/trainer.py

def run_federated_training(server, clients, rounds=20):
    for r in range(rounds):
        print(f"--- Round {r} ---")
        client_thetas = []
        client_sizes = []

        for c in clients:
            c.load_theta(server.global_theta)
            c.train_one_epoch()
            client_thetas.append(c.get_theta())
            client_sizes.append(len(c.dataloader.dataset))

        server.aggregate(client_thetas, client_sizes)
