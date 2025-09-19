import pandas as pd

print('\ndynamic_ordering_synthesis vs. pauliopt_steiner_gray')
df = pd.read_csv('results_molecules.csv')
#df = df.drop(columns=['u3', 'depth', '2q_depth'], axis = 1)
dfm = df[df.method == 'dynamic_ordering_synthesis']  
dfnm = df[df.method == 'pauliopt_steiner_gray']
#dfm = dfm.drop(columns=['method']).rename(columns={'time': 'mtime', 'cx': 'mcx'})
#dfnm = dfnm.drop(columns=['method']).rename(columns={'ntime': 'nmtime', 'cx': 'mcx'})
df = pd.merge(dfm, dfnm, on=['name', 'backend'], how='outer')
df = df.drop(columns=['num_qubits_y', 'n_gadgets_y', 'method_x', 'method_y'], axis=1)
df['cx diff %'] = (df['cx_x'] / df['cx_y']-1)*100
df['time diff %'] = (df['time_x'] / df['time_y']-1)*100
df = df.round({'cx diff %':2, 'time diff %':2, 'time_x':2, 'time_y':2})
df = df.rename(columns={
    'num_qubits_x': 'num_qubits',
    'n_gadgets_x': 'n_gadgets',
    'cx_y': 'cx sg',
    'cx_x': 'cx do',
    'time_y': 'time sg',
    'time_x': 'time do',
})
print(df.sort_values(['name']).to_string()) 


print('\ndynamic_ordering_synthesis_mapping vs. dynamic_ordering_synthesis')
df = pd.read_csv('results_molecules.csv')
#df = df.drop(columns=['u3', 'depth', '2q_depth'], axis = 1)
dfm = df[df.method == 'dynamic_ordering_synthesis_mapping']  
dfnm = df[df.method == 'dynamic_ordering_synthesis']
#dfm = dfm.drop(columns=['method']).rename(columns={'time': 'mtime', 'cx': 'mcx'})
#dfnm = dfnm.drop(columns=['method']).rename(columns={'ntime': 'nmtime', 'cx': 'mcx'})
df = pd.merge(dfm, dfnm, on=['name', 'backend'], how='outer')
df = df.drop(columns=['num_qubits_y', 'n_gadgets_y', 'method_x', 'method_y'], axis=1)
df['cx diff %'] = (df['cx_x'] / df['cx_y']-1)*100
df['time diff %'] = (df['time_x'] / df['time_y']-1)*100
df = df.drop(columns=['time_x', 'time_y'], axis=1)
df = df.round({'cx diff %':2, 'time diff %':2})
df = df.rename(columns={
    'num_qubits_x': 'num_qubits',
    'n_gadgets_x': 'n_gadgets',
    'cx_y': 'cx do',
    'cx_x': 'cx dom',
})
print(df.sort_values(['name']).to_string()) 

print('\ndynamic_ordering_synthesis_mapping vs. pauliopt_steiner_gray')
df = pd.read_csv('results_molecules.csv')
#df = df.drop(columns=['u3', 'depth', '2q_depth'], axis = 1)
dfm = df[df.method == 'dynamic_ordering_synthesis_mapping']  
dfnm = df[df.method == 'pauliopt_steiner_gray']
#dfm = dfm.drop(columns=['method']).rename(columns={'time': 'mtime', 'cx': 'mcx'})
#dfnm = dfnm.drop(columns=['method']).rename(columns={'ntime': 'nmtime', 'cx': 'mcx'})
df = pd.merge(dfm, dfnm, on=['name', 'backend'], how='outer')
df = df.drop(columns=['num_qubits_y', 'n_gadgets_y', 'method_x', 'method_y'], axis=1)
df['cx diff %'] = (df['cx_x'] / df['cx_y']-1)*100
df['time diff %'] = (df['time_x'] / df['time_y']-1)*100
df = df.drop(columns=['time_x', 'time_y'], axis=1)
df = df.round({'cx diff %':2, 'time diff %':2})
df = df.rename(columns={
    'num_qubits_x': 'num_qubits',
    'n_gadgets_x': 'n_gadgets',
    'cx_y': 'cx sg',
    'cx_x': 'cx dom',
})
print(df.sort_values(['name']).to_string()) 


print('\nall')
df = pd.read_csv('results_molecules.csv')
df1 = df[df.method == 'dynamic_ordering_synthesis']  
df2 = df[df.method == 'pauliopt_steiner_gray']
df3 = df[df.method == 'dynamic_ordering_synthesis_mapping']  
df = pd.merge(df1, df2, on=['name', 'backend'], how='outer')
df = df.drop(columns=['num_qubits_y', 'n_gadgets_y', 'method_x', 'method_y'], axis=1)
df = df.rename(columns={
    'num_qubits_x': 'num_qubits',
    'n_gadgets_x': 'n_gadgets',
    'cx_y': 'cx sg',
    'cx_x': 'cx do',
    'time_y': 'time sg',
    'time_x': 'time do',
})
df = pd.merge(df, df3, on=['name', 'backend'], how='outer')
df = df.drop(columns=['num_qubits_y', 'n_gadgets_y', 'method','time'], axis=1)
df = df.rename(columns={
    'num_qubits_x': 'num_qubits',
    'n_gadgets_x': 'n_gadgets',
    'cx': 'cx dom',
})
print(df.sort_values(['name']).to_string()) 
df.to_csv('results_molecules_all.csv')
