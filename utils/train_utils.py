
def generate_shells():
    with open('../shells/tmp.txt', 'a') as f:
        line = 'python train_vsgc.py --dataset cora --num_layers {}\n'
        for i in range(2, 51):
            command = line.format(i)
            f.write(command)


if __name__ == '__main__':
    generate_shells()