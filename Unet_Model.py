import numpy as np
import torch
from torch.nn import Conv3d, LeakyReLU, InstanceNorm3d
import network_blocks
import Data_Transform
import os

train_batch_size = 16
validation_batch_size = 8

def cartesian_product(*arrays):
    
    ndim = len(arrays)
    return np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, ndim)

slice_size = 64
stride = 32

input_dim = [slice_size, 64, slice_size]

x_slice_list = np.arange(0, (512 - slice_size + 1), stride)
z_slice_list = np.arange(0, (512 - slice_size + 1), stride)

transformation_selection_list = np.arange(0, 8)


class Unet_Model(torch.nn.Module):

    def .__init__(self):

        super(Unet_Model, self).__init__()
        self.blocks = network_blocks
        

    def forward(self, x_in):
        
        x_in_channels = int(x_in.shape[1])

        multi_elab_1 = self.blocks.Multiscale_Elab(x_in_channels)(x_in)

        multi_elab_1_channels = int(multi_elab_1.shape[1])
        multi_elab_1_1 = self.blocks.Multiscale_Elab(multi_elab_1_channels)(multi_elab_1) # To be used for concatenation

        multi_elab_1_1_channels = int(multi_elab_1_1.shape[1])

        reduction_1 = self.blocks.Reduction(multi_elab_1_1_channels)(multi_elab_1_1)

        reduction_1_channels = int(reduction_1.shape[1])

        multi_elab_2 = self.blocks.Multiscale_Elab(reduction_1_channels)(reduction_1)
        multi_elab_channels = int(multi_elab_2.shape[1])
        multi_elab_2_1 = self.blocks.Multiscale_Elab(multi_elab_channels)(multi_elab_2) # To be used for concatenation

        multi_elab_2_1_channels = int(multi_elab_2_1.shape[1])

        reduction_2 = self.blocks.Reduction(multi_elab_2_1_channels)(multi_elab_2_1)

        reduction_2_channels = int(reduction_2.shape[1])
        multi_elab_3 = self.blocks.Multiscale_Elab(reduction_2_channels)(reduction_2)
        multi_elab_3_channels = int(multi_elab_3.shape[1])
        multi_elab_3_1 = self.blocks.Multiscale_Elab(multi_elab_3_channels)(multi_elab_3) # To be used for concatenation

        multi_elab_3_1_channels = int(multi_elab_3_1.shape[1])
        reduction_3 = self.blocks.Reduction(multi_elab_3_1_channels)(multi_elab_3_1)
        reduction_3_channels = int(reduction_3.shape[1])

        
        multi_elab_4 = self.blocks.Multiscale_Elab(reduction_3_channels)(reduction_3)
        multi_elab_4_channels = int(multi_elab_4.shape[1])
        multi_elab_4_1 = self.blocks.Multiscale_Elab(multi_elab_4_channels)(multi_elab_4) # To be used for concatenation
        multi_elab_4_1_channels = int(multi_elab_4_1.shape[1])

        expansion_1 = self.blocks.Expansion(multi_elab_4_1_channels)(multi_elab_4_1)

        concat_1 = torch.cat([multi_elab_3_1, expansion_1], dim=1)
        concat_1_channels = int(concat_1.shape[1])

        multi_elab_5 = self.blocks.Multiscale_Elab(concat_1_channels)(concat_1)
        multi_elab_5_channels = int(multi_elab_5.shape[1])
        multi_elab_5_1 = self.blocks.Multiscale_Elab(multi_elab_5_channels)(multi_elab_5) # To be used for concatenation
        multi_elab_5_1_channels = int(multi_elab_5_1.shape[1])

        expansion_2 = self.blocks.Expansion(multi_elab_5_1_channels)(multi_elab_5_1)

        concat_2 = torch.cat([multi_elab_2_1, expansion_2], dim=1)
        concat_2_channels = int(concat_2.shape[1])

        multi_elab_6 = self.blocks.Multiscale_Elab(concat_2_channels)(concat_2)
        multi_elab_6_channels = int(multi_elab_6.shape[1])
        multi_elab_6_1 = self.blocks.Multiscale_Elab(multi_elab_6_channels)(multi_elab_6) # To be used for concatenation
        multi_elab_6_1_channels = int(multi_elab_6_1.shape[1])

        expansion_3 = self.blocks.Expansion(multi_elab_6_1_channels)(multi_elab_6_1)

        concat_3 = torch.cat([multi_elab_1_1, expansion_3], dim=1)
        concat_3_channels = int(concat_3.shape[1])

        multi_elab_7 = self.blocks.Multiscale_Elab(concat_3_channels)(concat_3)
        multi_elab_7_channels = int(multi_elab_7.shape[1])
        multi_elab_7_1 = self.blocks.Multiscale_Elab(multi_elab_7_channels)(multi_elab_7) # To be used for concatenation

        n_channel_in = multi_elab_7_1.shape[1]

        conv = Conv3d(in_channels=n_channel_in, out_channels=1, 
                    kernel_size=[1,1,1], stride=[1,1,1], padding=0, bias=True)
        
        torch.nn.init.xavier_uniform_(conv.weight)
        torch.nn.init.zeros_(conv.bias)

        conv_final = conv(multi_elab_7_1)

        return conv_final


    
model = Unet_Model()

device = torch.device("cuda")

model.to(device)

# Transfer Model to GPU

input_dim = [64, 64, 64]


batch_size = 4
in_channels = 2
input_data = torch.randn(batch_size, in_channels, input_dim[0], input_dim[1], input_dim[2])
input_data.to(device)

output = model(input_data)

print(output.shape)


transform = Data_Transform.transformation()

class data_pipeline(torch.utils.data.Dataset):

    def __init__(self, path, batch_size, dim, x_z_slice_size, stride, selection_list,
                 shuffle = True, n_channel = 2):
        
        self.dim= dim
        self.slice_size= x_z_slice_size
        self.stride= stride
        self.batch_size= batch_size
        self.selection_list = selection_list
        self.n_channels= n_channel
        self.path= path

        np.random.shuffle(self.selection_list)

    def __len__(self):
        
        elements = np.shape(self.selection_list)
        return int(np.floor(elements[0] / self.batch_size))
    
    def __getitem__(self,idx):

        'Generate a batch of data'
        indexes= self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        index_list_temp= [self.selection_list[k] for k in indexes]

        X_in, Y_in = self.__data_generator(index_list_temp)

        print(X_in.shape, Y_in.shape)

        return X_in, Y_in
        

    def __data_generator(self, selection_list):


        X = np.empty((self.batch_size, *self.dim, self.n_channels)) 
        Y = np.empty((self.batch_size, *self.dim, 1)) 

        for i, ID in enumerate(selection_list):

            CT_scan = np.load(self.path+'/CT_scans'+'/CT_'+ID[0]+'.npy').astype(np.float32)
            Dose_5K = np.load(self.path+'/Dose_5K'+'/Dose5K_'+ID[0]+'.npy').astype(np.float32)
            Dose_1M = np.load(self.path+'/Dose_1M'+'/Dose1M_'+ID[0]+'.npy').astype(np.float32)

            #Normalization

            CT_scan_norm = (CT_scan + 1000) / 21935
            Dose_5K_norm = (Dose_5K - np.min(Dose_5K)) / (np.max(Dose_5K) - np.min(Dose_5K))
            Dose_1M_norm = (Dose_1M - np.min(Dose_1M)) / (np.max(Dose_1M) - np.min(Dose_1M))

            CT_scan_slice = CT_scan_norm[ID[1]:ID[1]+slice,:, ID[2]:ID[2]+slice]
            Dose_5K_slice = Dose_5K_norm[ID[1]:ID[1]+slice,:, ID[2]:ID[2]+slice]
            Dose_1M_slice = Dose_1M_norm[ID[1]:ID[1]+slice,:, ID[2]:ID[2]+slice]
            
            if ID[3] == '1':
                CT_scan_slice = transform.flip(CT_scan_slice, 0)
                Dose_5K_slice = transform.flip(Dose_5K_slice, 0)
                Dose_1M_slice = transform.flip(Dose_1M_slice, 0)

            elif ID[3] == '2':
                CT_scan_slice = transform.flip(CT_scan_slice, 1)
                Dose_5K_slice = transform.flip(Dose_5K_slice, 1)
                Dose_1M_slice = transform.flip(Dose_1M_slice, 1)

            elif ID[3] == '3':
                CT_scan_slice = transform.flip(CT_scan_slice, 2)
                Dose_5K_slice = transform.flip(Dose_5K_slice, 2)
                Dose_1M_slice = transform.flip(Dose_1M_slice, 2)

            elif ID[3] == '4':
                CT_scan_slice = transform.rotation(CT_scan_slice, 1, (0, 2))
                Dose_5K_slice = transform.rotation(Dose_5K_slice, 1, (0, 2))
                Dose_1M_slice = transform.rotation(Dose_1M_slice, 1, (0, 2))

            elif ID[3] == '5':
                CT_scan_slice = transform.rotation(CT_scan_slice, 2, (0, 1))
                Dose_5K_slice = transform.rotation(Dose_5K_slice, 2, (0, 1))
                Dose_1M_slice = transform.rotation(Dose_1M_slice, 2, (0, 1))

            elif ID[3] == '6':
                CT_scan_slice = transform.rotation(CT_scan_slice, 2, (1, 2))
                Dose_5K_slice = transform.rotation(Dose_5K_slice, 2, (1, 2))
                Dose_1M_slice = transform.rotation(Dose_1M_slice, 2, (1, 2))


            X[i,]= np.stack((CT_scan_slice, Dose_5K_slice), axis=-1)
            Y[i,]= Dose_1M_slice.reshape((*self.dim, 1))



        X= torch.from_numpy(np.stack((CT_scan_slice, Dose_5K_slice), axis=0))
        Y= torch.from_numpy(Dose_1M_slice.reshape((*self.dim, 1)))

        return X, Y


train_path = '/NFSHOME/mspezialetti/sharedFolder/U-net_dataset/training'
validation_path = '/NFSHOME/mspezialetti/sharedFolder/U-net_dataset/validation'


training_examples = 785
index_list_train = [f'{num:03}' for num in range (1, training_examples+1)]
index_list_train = np.array(index_list_train)

train_resulting_list = cartesian_product(index_list_train, x_slice_list, z_slice_list, transformation_selection_list)

prams_train = {'selection_list': train_resulting_list,
                'batch_size':train_batch_size,
                'dim': input_dim,
                'x_z_slice_size': slice_size,
                'stride':stride
                }

validation_examples = 75
index_list_validation = [f'{num:03}' for num in range (1, validation_examples+1)]
index_list_validation = np.array(index_list_validation)

val_resulting_list = cartesian_product(index_list_validation, x_slice_list, z_slice_list, transformation_selection_list) 

prams_validation = {'selection_list': val_resulting_list,
                    'batch_size':validation_batch_size,
                    'dim':input_dim,
                    'x_z_slice_size':slice_size,
                    'stride':stride
                    }

training_set = data_pipeline(path=train_path, **prams_train)

validation_set = data_pipeline(path=validation_path, **prams_validation)

training_generator = torch.utils.data.DataLoader(training_set, batch_size=train_batch_size, 
                                                 shuffle=True, num_workers=10)

validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=train_batch_size, 
                                                   shuffle=False, num_workers=10)

criteria = torch.nn.MSELoss()
print(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       patience=4, factor=0.2, verbose=True)


def train_one_epoch (epoch_index):

    running_loss = 0.0
    last_loss = 0.0

    for batch_index, data in enumerate(training_generator):

        input, ref = data

        input.to(device)
        ref.to(device)

        optimizer.zero_grad()

        output = model.U_net(input)

        loss = criteria(output, ref)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_index % 10 == 9:

            last_loss = running_loss/10
            running_loss = 0.0

            # gotta work on tensorboard integration

    return last_loss


Epochs = 100
n_epoch = 0

for epoch in range(Epochs):

    print("starting epoch: ", n_epoch)

    model.train(True)
    avg_loss = train_one_epoch(epoch)

    running_vloss = 0.0

    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(validation_generator):

            v_input, v_output = vdata
            v_input.to(device)
            v_output.to(device)
            model_prediction = model(v_input)
            v_loss = criteria(model_prediction, v_output)
            running_vloss += v_loss.item()

        scheduler.step(running_vloss)

    n_epoch += 1
    
torch.save(model.state_dict(), '/NFSHOME/mjawaid/U_net/Pytorch_LMS/weights.pt')
