import boto3
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, array_contains, rand, rank, col
import time as tm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from datetime import timedelta
from pyspark.sql.window import Window

'''This file contains methods used to pull raw NFL tracking data into usable data formats for feature engineering and model building'''

def get_context_df(plist, elist, team):
    playlistdf = plist.filter((plist.play.playState =='APPROVED') & (plist.play.seasonType =='REG') & (plist.play.possessionTeam==team) & (plist.play.playType=='play_type_pass')).select(plist.play.gameId.alias('gameid'), plist.play.playId.alias('playid'))
    eventsdf = elist.filter((elist['los'].isNotNull()) & (elist['eventtype']=='ball_snap')).select(elist['gameid'], elist['playid'], elist['los'], elist['losside'], elist['qtr'], elist['time'].alias('ballsnaptime'))

    # Join playlist and events by playid
    playlist = playlistdf.alias('playlist')
    events = eventsdf.alias('events')

    gid_pid_matchcond=[playlist.playid==events.playid, playlist.gameid==events.gameid]
    contextdf = playlist.join(events, gid_pid_matchcond).select('playlist.gameid', 'playlist.playid', 'los', 'losside', 'qtr', 'ballsnaptime')

    # Capture only 3 passing plays per quarter in each game, randomize selection from quarter to remove time bias
    window = Window.partitionBy([contextdf['gameid'], contextdf['qtr']]).orderBy(rand())
    contextdf = contextdf.select('*', rank().over(window).alias('rank')).filter(col('rank') <= 3).limit(200)

    return contextdf

def o_d_separator(playerTrackingDF):
    # Explode and reduce player tracking data to only players involved in the play
    homeplayerdf_e1 = playerTrackingDF.select('gameId', 'startTime', 'endTime', explode('homeTrackingData').alias('homeplayerdata'))
    hp_df = homeplayerdf_e1.filter(homeplayerdf_e1.homeplayerdata.playerTrackingData[0].isOnField=='true')

    awayplayerdf_e1 = playerTrackingDF.select('gameId', 'startTime', 'endTime', explode('awayTrackingData').alias('awayplayerdata'))
    ap_df = awayplayerdf_e1.filter(awayplayerdf_e1.awayplayerdata.playerTrackingData[0].isOnField=='true')

    # Get only players where position group = o-line from homeplayerdata and awayplayer data - this is the o-line dataset for the selected team
    homeoffense = hp_df.filter((hp_df.homeplayerdata.positionGroup == 'OL')|(hp_df.homeplayerdata.positionGroup == 'QB')).select(hp_df.startTime, hp_df.endTime, hp_df.homeplayerdata.displayName.alias('player'), hp_df.homeplayerdata.position.alias('position'), hp_df.homeplayerdata.playerTrackingData.alias('trackingData'))
    awayoffense = ap_df.filter((ap_df.awayplayerdata.positionGroup == 'OL')|(ap_df.awayplayerdata.positionGroup == 'QB')).select(ap_df.startTime, ap_df.endTime, ap_df.awayplayerdata.displayName.alias('player'), ap_df.awayplayerdata.position.alias('position'), ap_df.awayplayerdata.playerTrackingData.alias('trackingData'))
    offense = homeoffense.union(awayoffense)

    # Get all defensive players, need to account for them being on home or away team
    homedefense = hp_df.filter((hp_df.homeplayerdata.positionGroup == 'DL')|(hp_df.homeplayerdata.positionGroup == 'DB')|(hp_df.homeplayerdata.positionGroup == 'LB')).select(hp_df.startTime, hp_df.endTime, hp_df.homeplayerdata.displayName.alias('player'),hp_df.homeplayerdata.position.alias('position'), hp_df.homeplayerdata.playerTrackingData.alias('trackingData'))
    awaydefense = ap_df.filter((ap_df.awayplayerdata.positionGroup == 'DL')|(ap_df.awayplayerdata.positionGroup == 'DB')|(ap_df.awayplayerdata.positionGroup == 'LB')).select(ap_df.startTime, ap_df.endTime, ap_df.awayplayerdata.displayName.alias('player'),ap_df.awayplayerdata.position.alias('position'), ap_df.awayplayerdata.playerTrackingData.alias('trackingData'))
    defense = homedefense.union(awaydefense)

    return {'offense': offense, 'defense': defense}

def join_context(offense, defense, context):
    # Join context df to tracking dfs using time of ball snap so timesteps can be filtered to events after the snap
    cdf = context.alias('cdf')

    # Offense data
    snapInPlayCond = [cdf.ballsnaptime > offense.startTime, cdf.ballsnaptime < offense.endTime]
    join1 = cdf.join(offense, snapInPlayCond).select('ballsnaptime', 'startTime', 'endTime', 'player', 'position', explode('trackingData').alias('timeaction'))

    # Defense data
    snapInPlayCond2 = [cdf.ballsnaptime > defense.startTime, cdf.ballsnaptime < defense.endTime]
    join2 = cdf.join(defense, snapInPlayCond2).select('ballsnaptime', 'startTime', 'endTime', 'player', 'position', explode('trackingData').alias('timeaction') )

    # Filter all timesteps before the snap for each play
    offensedata = join1.filter(join1['timeaction'].time >= join1['ballsnaptime']).select('ballsnaptime', 'player', 'position', join1.timeaction.time.alias('time'), join1.timeaction.x.alias('x'), join1.timeaction.y.alias('y'), join1.timeaction.dir.alias('dir'), join1.timeaction.o.alias('o'), join1.timeaction.s.alias('s'))
    defensedata = join2.filter(join2['timeaction'].time >= join2['ballsnaptime']).select('ballsnaptime', 'player', 'position', join2.timeaction.time.alias('time'), join2.timeaction.x.alias('x'), join2.timeaction.y.alias('y'), join2.timeaction.dir.alias('dir'), join2.timeaction.o.alias('o'), join2.timeaction.s.alias('s'))

    # Pandarize dataframes
    offense_df = offensedata.toPandas()
    defense_df = defensedata.toPandas()

    return {'offense': offense_df, 'defense': defense_df}

def prep_nfl_data(plist, elist, team):
    # Set up s3 bucket accessor and Spark Session
    s3 = boto3.resource('s3')
    bucket = s3.Bucket("nyg-hackathon-949955964069")
    spark = SparkSession \
        .builder \
        .appName("Spark SQL For NGS Dataset") \
        .config("spark.jars.packages", "com.amazonaws:aws-java-sdk:1.7.4,org.apache.hadoop:hadoop-aws:2.7.3") \
        .getOrCreate()
    
    contextdf = get_context_df(plist, elist, team)

    # Convert playdf to Pandas DataFrame for easier iterability to pull data
    context_df = contextdf.toPandas()
    
    playdatafiles = []
    for index, row in context_df.iterrows():
        for obj in bucket.objects.filter(Prefix='source_file/nfl/tracking_game_range/year=2019/tracking_game_range_' + str(row['gameid']) + '_' + str(row['playid'])):
            playdatafiles.append("s3a://nyg-hackathon-949955964069/" + obj.key)
            
    playerTrackingDF = spark.read.json(playdatafiles)

    offense = o_d_separator(playerTrackingDF)['offense']
    defense = o_d_separator(playerTrackingDF)['defense']

    # Join offense / defense to context df and pandarize dataframes (context_df already Pandarized)
    offense_df = join_context(offense, defense, contextdf)['offense']
    defense_df = join_context(offense, defense, contextdf)['defense']

    # Write tracking data to csv for feature generation from tracking data
    offense_df.to_csv('s3a://nyg-hackathon-949955964069/technicaldangerousness/SparkTables/' + team + '_tracking.csv')
    defense_df.to_csv('s3a://nyg-hackathon-949955964069/technicaldangerousness/SparkTables/' + team + '_trackingdefense.csv')
    
    ''' Analysis to determine player assignments '''
    # Create dynamic o-line dataframe containing player name, number of observations, and block wins
    olineplayerstats = offense_df.groupby('player').first()[['position']]
    olineplayerstats = olineplayerstats[olineplayerstats['position']!='QB']

    olineplayerstats['observations'] = 0
    olineplayerstats['blockwins'] = 0
    olineplayerstats['winrate'] = 0
    print('Before watching plays: ')
    print(olineplayerstats)

    # Iterate through plays in context_df and extract block win rate for each player
    counter = 0
    for index, play in context_df.iterrows():
        o_play = offense_df[offense_df['ballsnaptime']==play['ballsnaptime']]
        d_play = defense_df[defense_df['ballsnaptime']==play['ballsnaptime']]

        # Only first 2.5 seconds of play are relevant for our analysis. Note 2.5 seconds = 25 increments of time
        timesteps = o_play.groupby('time', as_index=False).first()['time']
        timesteps = timesteps[:25]
            
        o_play.loc[:,'x'] -= 10
        d_play.loc[:,'x'] -= 10
        
        # Get starting position of QB and Center to orient the game correctly
        try:
            start_center_pos = o_play[(o_play['position']=='C')&(o_play['time']==timesteps[0])]['x'].tolist()[0]
            start_qb_pos = o_play[(o_play['position']=='QB')&(o_play['time']==timesteps[0])]['x'].tolist()[0]
        except:
            continue
        
        # Determine direction of offensive play
        if start_center_pos > start_qb_pos:
            left_to_right = True
        else:
            left_to_right = False
        
        # Determine absolute x of LOS
        if start_center_pos < 50:
            los_x = play['los']
        else:
            los_x = 100 - play['los']

        # Initialize passrushers list: define as defenders that have crossed the line of scrimmage by the end of obs. window
        passrushers = []
        for index, player in d_play.iterrows():
            # Define 'crossing LOS' depending on which direction the offense is going
            if left_to_right == True:
                crossed_los = (player['x'] <= los_x)
            else:
                crossed_los = (player['x'] >= los_x)
            
            if crossed_los and player['player'] not in passrushers:
                passrushers.append(player['player'])
        
        if len(passrushers) < 4:
            print('Faulty passrushers encountered')
            continue
                
        # Initialize assignments df - get all players except for QB
        assignments = o_play.groupby('player', as_index=False).first()[['player', 'position']]
        assignments = assignments[assignments['position'] != 'QB']
        assignments['assignment'] = 'None'
        
        # Dynamically assign coverage using distance and orientation wrt each pass rusher at each time step
        # playerlosstracker will reflect 0 for a player if player was never beaten by his assignment
        playerlosstracker = {}

        # Every time the below for loop is run, 
        # 1. Increment 'observations' for each player observed by 1
        # 2. Reset playerlosstracker to 0 to observe the new play
        for index, row in assignments.iterrows():
            obs_gain = olineplayerstats[olineplayerstats.index==row['player']]['observations']+1
            olineplayerstats.at[row['player'], 'observations'] = obs_gain
            playerlosstracker[row['player']] = 0
        
        for time in timesteps:
            
            # Set a Boolean flag - if continue1 conditions are met, continue1 is set to True and all loops within this are exited, effectively skipping to the next time step. This is used to deal with missing data
            continue1 = False
            
            # Data to analyze in each iteration
            d_step = d_play[d_play['time']==time]
            o_step = o_play[o_play['time']==time]
                        
            # Get QB Position
            qb_xy = np.array([o_step[o_step['position']=='QB']['x'].iloc[0],o_step[o_step['position']=='QB']['y'].iloc[0]])
            
            # Loop through each player, get their x,y coordinates, and compare to each pass rusher
            for index, row in assignments.iterrows():
                try:
                    ol_xy = np.array([o_step[o_step['player']==row['player']]['x'].iloc[0], o_step[o_step['player']==row['player']]['y'].iloc[0]]) 
                
                # Bad programming solution to the problem of missing data for time's sake, but using this escape flag to skip time step if data is missing for a player
                except:
                    continue1 = True
                    continue
                
                distances_to_rushers = {}

                for name in passrushers:
                    d_xy = np.array([d_step[d_step['player']==name]['x'].iloc[0], d_step[d_step['player']==name]['y'].iloc[0]])
                    distance = np.linalg.norm(d_xy-ol_xy)
                    distances_to_rushers.update({name: distance})
                
                assignment = min(distances_to_rushers, key=distances_to_rushers.get)
                row['assignment'] = assignment

                # Track coordinates of assignment specifically
                asn_xy = np.array([d_step[d_step['player']==assignment]['x'].iloc[0], d_step[d_step['player']==assignment]['y'].iloc[0]])

                # Key calculation: if assignment is ever closer to QB than o-lineman and < 3 yards away in 2.5 s, set Boolean win = False
                if np.linalg.norm(asn_xy-qb_xy) < np.linalg.norm(ol_xy-qb_xy) and np.linalg.norm(asn_xy-qb_xy) < 4:
                    playerlosstracker[row['player']] += 1 
            
            if continue1 == True:
                continue
        
        for player in playerlosstracker:
            if playerlosstracker[player] == 0:
                win_gain = olineplayerstats.loc[player, 'blockwins'] + 1
                olineplayerstats.at[player, 'blockwins'] = win_gain
        counter+=1    

    olineplayerstats['winrate'] = olineplayerstats['blockwins']/olineplayerstats['observations']
    olineplayerstats = olineplayerstats[olineplayerstats['observations'] > 50]
    olineplayerstats.to_csv('s3a://nyg-hackathon-949955964069/technicaldangerousness/SparkTables/' + team + '_playerstats.csv')
    
    print('After watching play:')
    print(olineplayerstats)
    
    return True