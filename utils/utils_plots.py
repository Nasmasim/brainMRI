import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss_vs_scores(num_epochs, 
                        loss_train_log, epoch_val_log, loss_val_log, 
                        bg_dice_log, gm_dice_log, wm_dice_log, csf_dice_log):
    # plot the loss 
    plt.plot(range(1, num_epochs + 1), loss_train_log, c='r', label='train')
    plt.plot(epoch_val_log, loss_val_log, c='b', label='val')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    
    # plt the dice score
    plt.plot(epoch_val_log, bg_dice_log, label='bg_dice')
    plt.plot(epoch_val_log, gm_dice_log, label='gm_dice')
    plt.plot(epoch_val_log, wm_dice_log, label='wm_dice')
    plt.plot(epoch_val_log, csf_dice_log, label='csf_dice')
    plt.legend(loc='upper right')
    plt.title('Dice Score')
    plt.xlabel('epoch')
    plt.ylabel('Dice score')
    plt.show()
    
def plot_volume_vs_age(vols, meta_data_reg_train, title):
    plt.scatter(vols[0,:],meta_data_reg_train['age'], marker='.')
    plt.scatter(vols[1,:],meta_data_reg_train['age'], marker='.')
    plt.scatter(vols[2,:],meta_data_reg_train['age'], marker='.')
    plt.grid()
    plt.title(title)
    plt.xlabel('Volume')
    plt.ylabel('Age')
    plt.legend(('CSF','GM','WM'))
    plt.show()
    
def display_statistics (meta_data_all):
    meta_data = meta_data_all
    
    sns.catplot(x="gender_text", data=meta_data, kind="count")
    plt.title('Gender distribution')
    plt.xlabel('Gender')
    plt.show()
    
    sns.distplot(meta_data['age'], bins=[10,20,30,40,50,60,70,80,90])
    plt.title('Age distribution')
    plt.xlabel('Age')
    plt.show()
    
    plt.scatter(range(len(meta_data['age'])),meta_data['age'], marker='.')
    plt.grid()
    plt.xlabel('Subject')
    plt.ylabel('Age')
    plt.show()