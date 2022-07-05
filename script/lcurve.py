def lcurve(hists, colors,epochs) :
  import matplotlib.pyplot as plt
  plt.figure(figsize=(12,4))
  # loss
  plt.subplot(121)
  for i,hist in enumerate(hists) :
    loss = eval(hist).history['loss']
    val_loss = eval(hist).history['val_loss']
    plt.plot(epochs, loss, linestyle = ':', label = f'{hist} train_loss', c = colors[i])
    plt.plot(epochs, val_loss, marker = '.', label = f'{hist} val_loss', c = colors[i])
    plt.title('Loss')
    plt.legend();plt.grid(True);plt.xticks(epochs)
    plt.xlabel('Epochs');plt.ylabel('Loss')
    x,y = epochs[-1], eval(hist).history['loss'][-1]
    plt.text(x,y,np.round(y,2), c=colors[i])
    x,y = epochs[-1], eval(hist).history['val_loss'][-1]
    plt.text(x,y,np.round(y,2), c=colors[i])

  # acc
  plt.subplot(122)
  for i,hist in enumerate(hists) :
    acc = eval(hist).history['acc']
    val_acc = eval(hist).history['val_acc']
    plt.plot(epochs,acc, linestyle = ':', label = f'{hist} train_acc', c = colors[i])
    plt.plot(epochs, val_acc, marker = '.', label = f'{hist} val_acc', c = colors[i])
    plt.title('Acc')
    plt.legend();plt.grid(True);plt.xticks(epochs)
    plt.xlabel('Epochs');plt.ylabel('Acc')
    x,y = epochs[-1], eval(hist).history['acc'][-1]
    plt.text(x,y,np.round(y,2), c=colors[i])
    x,y = epochs[-1], eval(hist).history['val_acc'][-1]
    plt.text(x,y,np.round(y,2), c=colors[i])  
  plt.show()