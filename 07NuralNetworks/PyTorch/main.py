import torch

def __say_hello_pytorch(name):
    print(f"{'-'*20}")
    print(f'Hi Pytorch from: {name}')
    print(torch.__version__)
    print(torch.cuda.is_available())  # True if CUDA (GPU) is accessible
    x = torch.rand(3, 3)
    print("Random Tensor:\n", x)


def __collections_pytorch_operations(lst, tpl, __set):
    print(f"{'-' * 20}")
    print(f"List and Tuple Operations\n List={lst} :: Tuple={tpl}")
    lst[1] = 'changed'
    print(f" Changed List={lst} :: Tuples are immutable, trying to change them gives error")
    print(f" Iterate over Collections")
    for i in tpl:
        print(f'  {i}')
    print(f" Indexing through 'enumerate'")
    for index_of_tuple, tuple_item in enumerate(tpl):
        print(f'  {index_of_tuple} : {tuple_item}')
    print(f" Append:")
    lst.append('c')         # set cannot have duplicates

    for index_of_list, list_item in enumerate(lst):
        print(f'  {index_of_list} : {list_item}')
    print(f" Printing Set:")
    for i in __set:
        print(f"  {i}")
    __set.append('b')    # set cannot have duplicates, set does not have append() method


if __name__ == '__main__':
    # __say_hello_pytorch('Anodiam')
    __collections_pytorch_operations(['a', 'b', 'c', 'd'],
                                     ('a', 'b', 'c', 'd'),
                                     {'a', 'b', 'c', 'd', 'c'})