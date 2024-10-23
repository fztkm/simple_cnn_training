CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -d ImageFolder -r /home/fukuzawa/dataset/places365/places365_standard/ -w 8 -b 128 -e 4 --optimizer SGD -m clip -lr 1e-3 --debug
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -d ImageFolder -r /home/fukuzawa/dataset/places365/places365_standard/ -w 8 -b 128 -e 1 --optimizer SGD -m clip -lr 2e-2 --debug
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -d ImageFolder -r /home/fukuzawa/dataset/places365/places365_standard/ -w 8 -b 128 -e 1 --optimizer SGD -m clip -lr 4e-2 --debug
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -d ImageFolder -r /home/fukuzawa/dataset/places365/places365_standard/ -w 8 -b 128 -e 2 --optimizer SGD -m clip -lr 6e-2 --debug
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -d ImageFolder -r /home/fukuzawa/dataset/places365/places365_standard/ -w 8 -b 128 -e 1 --optimizer SGD -m clip -lr 7e-3 --debug
wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -d ImageFolder -r /home/fukuzawa/dataset/places365/places365_standard/ -w 8 -b 128 -e 1 --optimizer Adam -m clip -lr 5e-3
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -d ImageFolder -r /home/fukuzawa/dataset/places365/places365_standard/ -w 8 -b 128 -e 1 --optimizer SGD -m clip -lr 8e-3
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -d ImageFolder -r /home/fukuzawa/dataset/places365/places365_standard/ -w 8 -b 128 -e 1 --optimizer Adam -m clip -lr 8e-3
# wait
