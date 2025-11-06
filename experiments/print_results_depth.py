import pandas as pd

print('\nshortest_path_in_pauli_forest vs. pauliopt_steiner_gray')
df = pd.read_csv('results_molecules.csv')
#df = df.drop(columns=['u3', 'depth', '2q_depth'], axis = 1)
dfm = df[df.method == 'shortest_path_synthesis']  
dfnm = df[df.method == 'pauliopt_steiner_gray']
#dfm = dfm.drop(columns=['method']).rename(columns={'time': 'mtime', 'cx': 'mcx'})
#dfnm = dfnm.drop(columns=['method']).rename(columns={'ntime': 'nmtime', 'cx': 'mcx'})
df = pd.merge(dfm, dfnm, on=['name', 'backend'], how='outer')
df = df.drop(columns=['num_qubits_y', 'n_gadgets_y', 'method_x', 'method_y','cx_x','cx_y'], axis=1)
df['cx diff %'] = (df['cx-depth_x'] / df['cx-depth_y']-1)*100
df['time diff %'] = (df['time_x'] / df['time_y']-1)*100
df = df.round({'cx diff %':2, 'time diff %':2, 'time_x':2, 'time_y':2})
df = df.rename(columns={
    'num_qubits_x': 'num_qubits',
    'n_gadgets_x': 'n_gadgets',
    'cx-depth_y': 'cx lo',
    'cx-depth_x': 'cx sp',
    'time_y': 'time lo',
    'time_x': 'time sp',
})
print(df.sort_values(['name']).to_string()) 

df.to_csv('results_molecules_all.csv')
