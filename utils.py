from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

def get_geo_list(df):
    '''
    Get the latitude and longitude list from the dataframe
    '''
    lats, lons = df['latitude'].tolist(), df['longitude'].tolist()
    return lats, lons

def sample_by_state(df, samples):
    '''
    Sample the dataframe by state(to avoid clustered pts on map)
    '''
    df_sampled = df.groupby('state').apply(
        lambda x: x.sample(samples, replace = True)).reset_index(drop = True)
    return df_sampled
    

def plot_geomap(df, col, task_name, by_state = True, 
                savefig = False, samples = 20000):
    '''
    Plot the geographical map with color coded by the metric'''

    lats, lons, metric_list = [],[],[]
    
    # the asos_stations file can be found here: 
    # https://engineersportal.com/s/asos_stations.csv

    if(len(df) > 50000):
        df = sample_by_state(df, samples // 50)
    lats, lons = get_geo_list(df)
    if(by_state):
        #get state level stats
        state_df = df.groupby('state')[col].mean()
        metric_list = df['state'].apply(
            lambda s: state_df[s].item()).tolist()
    else:
        metric_list = df[col].tolist()    
    

    print(f'num data = {len(metric_list)},null = {df[col].isnull().sum()}')
    print(np.array(metric_list).min(), np.array(metric_list).max())

    # How much to zoom from coordinates (in degrees)
    zoom_scale = 0

    # Setup the bounding box for the zoom and bounds of the map
    # bbox = [np.min(lats)-zoom_scale,47.16+zoom_scale,\
    #         -130-zoom_scale,-57+zoom_scale]
    bbox = [np.min(lats)-zoom_scale,np.max(lats)+zoom_scale,\
        np.min(lons)-zoom_scale,np.max(lons)+zoom_scale]
    print(f'map boundary = {bbox}')

    fig, ax = plt.subplots(figsize=(12,7))
    plt.title(f"US Regional Distribution on {task_name}")
    # Define the projection, scale, the corners of the map, and the resolution.
    m = Basemap(projection='merc',llcrnrlat=bbox[0],urcrnrlat=bbox[1],\
                llcrnrlon=bbox[2],urcrnrlon=bbox[3],lat_ts=10,resolution='i')

    # Draw coastlines and fill continents and water with color
    m.drawcoastlines()
    m.fillcontinents(color='black',lake_color='lightblue')
    #m.drawlsmask(land_color='#CCCCCC',ocean_color='lightblue',lakes=True)

    # draw parallels, meridians, and color boundaries
    m.drawparallels(np.arange(bbox[0],bbox[1],(bbox[1]-bbox[0])/5),labels=[1,0,0,0])
    m.drawmeridians(np.arange(bbox[2],bbox[3],(bbox[3]-bbox[2])/5),labels=[0,0,0,1],rotation=15)
    m.drawmapboundary(fill_color='lightblue')

    # format colors for elevation range
    metric_min = np.min(metric_list)
    metric_max = np.max(metric_list)

    #metric_min, metric_max = -0.2, 0.2 #diff mean

    cmap = plt.get_cmap('jet')
    normalize = matplotlib.colors.Normalize(vmin=metric_min, vmax=metric_max)

    # plot values with different colors with numpy interpolate mapping tool
    for ii in range(0,len(metric_list)):
        x,y = m(lons[ii],lats[ii])
        color_interp = np.interp(metric_list[ii],[metric_min,metric_max],[50,200])
        color = cmap(int(color_interp))
        #color = 'green' if metric_list[ii]==1 else 'red'
        plt.plot(x,y,marker='o',markersize=0.1,color=color)

    # format the colorbar 
    cax, _ = matplotlib.colorbar.make_axes(ax)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,norm=normalize,label=col)

    # save the figure and show it
    if(savefig):
        plt.savefig(f'{task_name}_geomap.png', format='png',facecolor="white", dpi=500,transparent=True)
    plt.show()