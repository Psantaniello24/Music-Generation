import torch
import torch.nn as nn
import librosa
import soundfile as sf

#Define the diffusion generative model
class DiffusionModel(nn.Module):
  def __init__(self):
    super(DiffusionModel,self).__init__()
    self.conv1 = nn.Conv1d( 1 , 64 , kernel_size = 5 , stride = 2, padding = 2 )
    self.conv2 = nn.Conv1d( 64 , 128 , kernel_size = 5 , stride = 2 , padding = 2)
    self.fc = nn.Linear( 128 *  22050 , 128 *  22050)
    self.conv_transpose1 = nn.ConvTranspose1d(128,64, kernel_size = 5, stride = 2, padding = 2)
    self.conv_transpose2 = nn.ConvTranspose(64,1,kernel_size =5, stride = 2, padding = 2 )
  def forward(self,x):
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    print(f"Shape before .view : {x.shape}")
    #with view we flatten the shape of x
    x = x.view(x.size(0),-1)
    x = torch.relu(self.fc(x))
    x = x.view(x.size(0),128,-1)
    x = torch.relu(self.conv_transpose1(x))
    print(f"Shape after .view : {x.shape}")
    return torch.tanh(self.conv_transpose1(x))

model = DiffusionModel()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

#load a real song
real_music , sr = librosa.load('old-school-bass-beat-fur-djs-abmischen-rappen-110775.wav', sr = None)
real_music = torch.from_numpy(real_music).float().unsqueeze(0).unsqueeze(0)

#training loop
for epoch in range(100):

  generated_music = model(real_music)

  loss = criterion(generated_music,real_music)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  print(f"Epoch {epoch +1}, Loss : {loss.item()}")

#save generated song
generated_music = model(real_music)

#convert the tensor back to a numpy array
generated_music_np = generated_music.detach().numpy()


#write out audio file
sf.write('generated_song.wav', generated_music_np, sr )

print(real_music.shape)
