import numpy as np

arr1D = np.random.randint(0, 10, 30)
print(f'\nARRAY = {arr1D}\nDim = {arr1D.ndim} :: Size = {arr1D.size} :: Shape = {arr1D.shape} :: Dimensions = {arr1D.shape[0]} X UNDEFINED '
      f'\n** 1-DIM ARRAYS ARE TO BE AVOIDED GOING FORWARD **\n')

arr1Da = arr1D.reshape(1, 30)
print(f'1-DIM ARRAYS TO BE CONVERTED INTO ROW MATRIX = {arr1Da}'
      f'\nDim = {arr1Da.ndim} :: Size = {arr1Da.size} :: Shape = {arr1Da.shape} :: Rows = {arr1Da.shape[0]} :: Columns = {arr1Da.shape[1]}\n')

rowArr = np.array([[1, 2, 3, 4]])
print(f'ROW MATRIX = {rowArr}\nDim = {rowArr.ndim} :: Size = {rowArr.size} :: Shape = {rowArr.shape} :: '
      f'Rows = {rowArr.shape[0]}, Columns = {rowArr.shape[1]}\n')

colArr = rowArr.T
print(f'COLUMN MATRIX =\n{colArr}\nDim = {colArr.ndim} :: Size = {colArr.size} :: Shape = {colArr.shape} :: '
      f'Rows = {colArr.shape[0]}, Columns = {colArr.shape[1]}\n')

arr2D = np.reshape(arr1D, (5, 6))
print(f'2D-ARRAY =\n{arr2D}\nDim = {arr2D.ndim} :: Size = {arr2D.size} :: Shape = {arr2D.shape} :: '
      f'Rows = {arr2D.shape[0]}, Columns = {arr2D.shape[1]}\n')

arr2Da = np.reshape(arr2D, (3, 2, 5))
print(f'3D-ARRAY =\n{arr2Da}\nDim = {arr2Da.ndim} :: Size = {arr2Da.size} :: Shape = {arr2Da.shape} :: '
      f'Pages = {arr2Da.shape[0]}, Rows = {arr2Da.shape[1]}, Columns = {arr2Da.shape[2]}')

arr2Db = np.reshape(arr2Da, (1, 30))
print(f'\nReshape back to Row Matrix = {arr2Db}\nDim = {arr2Db.ndim} :: Size = {arr2Db.size} :: Shape = {arr2Db.shape} :: '
      f'Rows = {arr2Db.shape[0]}, Columns = {arr2Db.shape[1]}')