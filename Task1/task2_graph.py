import matplotlib.pyplot as plt
val_loss = [3.1124, 1.7907, 1.6172, 1.7131, 1.8646, 1.1362, 1.5945, 1.9756, 1.4576, 1.6259]
loss = [3.5314, 2.1463, 1.9560, 1.8561, 1.7876, 1.7006, 1.7058, 1.7189, 1.5919, 1.5753]
epoch = [1,2,3,4,5,6,7,8,9,10]

plt.plot(epoch, val_loss, label ='validation loss')

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss")
plt.legend()
plt.show()

plt.plot(epoch, loss,  label ='loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.show()