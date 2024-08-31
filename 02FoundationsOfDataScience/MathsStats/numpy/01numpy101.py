import numpy as np

# Array initialization
# arr1 = np.array([2, 5, 6, 8, 5, 2, 8])
# print(f'type of [2, 5, 6, 8, 5, 2, 8]= {type([2, 5, 6, 8, 5, 2, 8])}')
# print(f"arr1 = {arr1} :: Type = {arr1.dtype} :: ndim = {arr1.ndim} :: ** SHAPE = {arr1.shape} **")

# # Accessing array element same as accessing list element (zero based indexing)
# print(f"arr1[2] = {arr1[2]}")

# myList = [1, 2, 3, 4, 5, 6, 7]
# arr2 = np.array(myList)
# print(f"arr2 = {arr2}")

# arrbln = np.array([False, True, False, False, True, True, True, True, False])
# print(f"arrbln = {arrbln} :: Type = {arrbln.dtype}")

# arrstr = np.array(["asfd", "bvsfgbsg dfsf", "csvdvsfd  df f", "d", "e", "f", "g"])
# print(f"arrstr = {arrstr} :: Type = {arrstr.dtype}")

# arr2d = np.array([[.3, .4, 0], [.2, .9, .1], [.5, .1, .1]])
# print(f"arr2d\n{arr2d}\nType = {arr2d.dtype}")

# # Accessing array element same as accessing list element (zero based indexing)
# print(f"arr2d[2] = {arr2d[2]}")
# print(f"arr2d[2][0] = {arr2d[2][0]}")

# # Array properties: shape, size, dimensions, data type, item size, bytes, strides, base
# arr_my2d = np.array([[1, .3, .4, 0], [.2, False, .9, .1], [.5, .5, .1, .1]])
# print(f"arr_my2d = {arr_my2d}")
# print(f"** SHAPE :: arr_my2d.shape = {arr_my2d.shape} **")
# print(f"arr_my2d.size = {arr_my2d.size}")
# print(f"arr_my2d.ndim = {arr_my2d.ndim}")
# print(f"arr_my2d.dtype = {arr_my2d.dtype}")
# print(f"arr_my2d.itemsize = {arr_my2d.itemsize}")
# print(f"arr_my2d.nbytes = {arr_my2d.nbytes}")
# print(f"arr_my2d.strides = {arr_my2d.strides}")
# print(f"arr_my2d.data = {arr_my2d.data}")
# print(f"arr_my2d.base = {arr_my2d.base}")

# # What will happen if you try to create an array with elements of MIXED TYPE?
# arr_curious = np.array([[1, .3, .4, 0], [.2, False, .9, .1], [.5, .5, 'Wow Aniran!', .1]])
# print(f"arr_curious = {arr_curious}")
# print(f"arr_curious.shape = {arr_curious.shape}")
# print(f"arr_curious.size = {arr_curious.size}")
# print(f"arr_curious.ndim = {arr_curious.ndim}")
# print(f"arr_curious.dtype = {arr_curious.dtype}")

# # Other ARRAY Creations:
# arrng1 = np.arange(5)
# print(f"arrng1 = {arrng1}")

# arrng2 = np.arange(10, 15)          # [x, y)
# print(f"arrng2 = {arrng2}")

# arrng_step = np.arange(1, 15, 4)    # [x, y) step z
# print(f"arrng_step = {arrng_step}")

# zarr = np.zeros(15)
# print(f"zarr = {zarr}")

# # Library, Bookshelf, Rack, Book, Pages, rows and columns like reading data in a book
# 4 pages, 3 rows, 5 columns
# multi_zarr = np.zeros((4, 3, 5))
# print(f"multi_zarr\n{multi_zarr}")

# onearr = np.ones(5)
# print(f"onearr = {onearr}")

# eyearr = np.eye(5)
# print(f"eyearr\n{eyearr}")

# fullarr = np.full((3, 5), 'b')
# print(f"fullarr \n{fullarr}")

# linspace() divides the stop - start in equal gaps of specified number
# linspc_arr = np.linspace(1, 10, 7)
# print(f"linspc_arr = {linspc_arr} :: Array Length = {len(linspc_arr)} is specified\ngap = {linspc_arr[1] - linspc_arr[0]} :: Both {linspc_arr[0]} and {linspc_arr[-1]} are included")

# # Difference between arange and linspace
# arnge = np.arange(1, 10, 1.5)       # [ )
# print(f"arnge = {arnge} :: Array Length = {len(arnge)} is NOT specified\ngap = {linspc_arr[1] - linspc_arr[0]} IS specified :: {linspc_arr[-1]} is NOT included")

# geomspace() divides the stop - start into specified number of intervals of the same ratio
# geomspc_arr = np.geomspace(1, 10, 7) # Start = 0 gives ERROR
# print(f"geomspc_arr = {geomspc_arr} :: Array Length = {len(geomspc_arr)} is specified\nBoth {geomspc_arr[0]} and {geomspc_arr[-1]} are included\nRatio at last = {geomspc_arr[-1]/geomspc_arr[-2]}\nRatio at middle= {geomspc_arr[-3] / geomspc_arr[-4]}\nRatio at first= {geomspc_arr[1] / geomspc_arr[0]}")

# logspace() generates series of 10^start - 10^stop with exponents in number of intervals
# logspc_arr = np.logspace(1, 10, 7)
# print(f"logspc_arr = {logspc_arr} :: Array Length = {len(logspc_arr)} is specified\nBoth {(logspc_arr[0])} and {(logspc_arr[-1])} are included")

# # More on Shapes of 1d Arrays
# arr1d = np.random.randint(0,  10, 6)
# print(f"arr1d = {arr1d} :: ** SHAPE = {arr1d.shape} **")
# # arr1dT = arr1d.transpose()
# arr1dT = arr1d.T
# print(f"arr1dT = {arr1dT} :: ** SHAPE = {arr1dT.shape} **")

# arr2d = np.random.randint(0,  10, (1, 5))
# print(f"arr2d = {arr2d} :: ** SHAPE = {arr2d.shape} **")
# arr2dT = arr2d.T
# print(f"arr2dT =\n {arr2dT}\n:: ** SHAPE = {arr2dT.shape} **\nWe never use 1D arrays of shape (n, ) or ( ,n)\nEven for Vectors or 1 Dimensional Series of Numbers:\nWe use 2D Matrix as a ROW Vector of shape (1,n) or Column Vector of shape (n, 1)")
