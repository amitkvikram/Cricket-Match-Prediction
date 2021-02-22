import pandas as pd
from preprocessing import preprocessingHelper
import matplotlib.pyplot as plt
from matplotlib import animation
from model import get_model

def simulate(match_id):
    test_df = pd.read_csv("./data/test_data.csv", index_col = 0)
    test_df, test_df_c = preprocessingHelper.preprocess(test_df)

    match_df = (test_df[test_df['match-id'] == match_id]).reset_index()
    match_df_c = (test_df_c[test_df_c['match-id'] == match_id]).reset_index()
    X, Y = preprocessingHelper.getX_Y(match_df)
    
    battingTeam = match_df_c.loc[0, 'team-batting']
    bowlingTeam = match_df_c.loc[0, 'team-bowling']
    winner = match_df_c.loc[0, 'winner']
    target = match_df_c['target-score'][0]
    
    lstm_model = get_model()
    lstm_model.load_weights("weights.best.hdf5")
    p = lstm_model.predict(X)
    p_l = p.flatten().tolist()
    
    match = match_df_c
    fig=plt.figure(figsize=(12, 5))
    fig.suptitle(battingTeam + " vs " + bowlingTeam + ": " + str(match['date'][0]) +
                 " | " + "Winner: " + match['winner'][0] + "\n", 
                 fontsize=16, y=1.08)
    
    ax_1=fig.add_subplot(1,3,1)
    wicket_l = (10-match['wicket-left'].values.flatten()).tolist()
    runsScored_l = match['runs-scored'].values.flatten().tolist()
    overs_l = [str(x // 6)+"." + str(x%6) for x in match['balls-bowled'].values.flatten().tolist()]
    ax_1.axes.get_yaxis().set_visible(False)
    ax_1.axes.get_xaxis().set_visible(False)
    ax_1.set_frame_on(False)
    txt= ax_1.text(0.1, 0.5,'Run-Wickets | Overs', horizontalalignment='left',verticalalignment='center',transform = ax_1.transAxes, fontsize=18, 
                 bbox=dict(facecolor='white', edgecolor='white'))
    txt.set_clip_on(False)
    
    ax_2=fig.add_subplot(1,3,2)
    barcollection_2 = ax_2.bar([battingTeam, bowlingTeam], [p_l[0], 1.0 - p_l[0]])
    ax_2.set_ylim([0, 1.0])
    ax_2.set_ylabel("Probability", fontsize=14)
    ax_2.set_frame_on(False)
    
    ax_3=fig.add_subplot(1,3,3)
    reqRunRate_l = ((match['target-score'] - match['runs-scored'])/(120 - match['balls-bowled'])).values.tolist()
    currRunRate_l = (match['runs-scored']/(match['balls-bowled']+1.0)).values.tolist()
    barcollection_3 = ax_3.bar(["RRR", "CRR"], [reqRunRate_l[0]*6, currRunRate_l[0]*6])
    ax_3.set_ylim([0, 36])
    ax_3.set_ylabel("Runs", fontsize=14)
    ax_3.set_frame_on(False)
    plt.tight_layout()

    def animate(i):
        s = "Target: " + str(match['target-score'][0]) + "\n" + battingTeam + ": " + str(runsScored_l[i]) + "-" + str(wicket_l[i]) + " | " + overs_l[i]
        txt.set_text(s)
                     
        h_2 = [p_l[i], 1-p_l[i]]
        h_3 = [reqRunRate_l[i]*6, currRunRate_l[i]*6]
        for i, b in enumerate(barcollection_2):
            b.set_height(h_2[i])
        for i, b in enumerate(barcollection_3):
            b.set_height(h_3[i])

    anim=animation.FuncAnimation(fig, animate, repeat=False, blit=False,frames=len(wicket_l),
                                 interval=1000)
    
    anim.save("simulate"+str(match_id)+".mp4",writer=animation.FFMpegWriter(fps=2))