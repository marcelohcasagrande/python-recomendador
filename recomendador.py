# Importando pacotes.
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler # Padronização dos dados.
from sklearn.decomposition import PCA # Análise de Componentes Principais.
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import euclidean_distances # Distância Euclidiana.

# Lendo banco de dados.
dados = pd.read_csv( 'datasets/dados_totais.csv' )
dados_generos = pd.read_csv( 'datasets/data_by_genres.csv' )
dados_anos = pd.read_csv( 'datasets/data_by_year.csv' )


    #              #
    # Dados Totais #
    #              #

# Head.
dados.head()

# Info.
dados.info()

# Anos únicos.
dados[ 'year' ].unique()

# Dropando colunas desnecessárias.
dados.drop( [ 'explicit', 'key', 'mode' ], axis = 1, inplace = True )

# Conferindo se dropou.
dados.shape

# Conferindo se tem missings.
dados.isna().sum()


    #               #
    # Dados Gêneros #
    #               #

# Olhando dados gêneros.
dados_generos.head()  

# Dropando colunas desnecessárias.
dados_generos.drop( [ 'key', 'mode' ], axis = 1, inplace = True )

# Conferindo se tem missings.
dados_generos.isna().sum()


    #            #
    # Dados Anos #
    #            #

# Olhando dados anos.
dados_anos.head()  

# Anos únicos.
dados_anos[ 'year' ].unique()

# Limitando anos da base de anos.
dados_anos = dados_anos[ dados_anos[ 'year' ] >= 2000 ] 

# Anos únicos.
dados_anos[ 'year' ].unique()

# Dropando colunas desnecessárias.
dados_anos.drop( [ 'key', 'mode' ], axis = 1, inplace = True )

# Resetando index.
dados_anos.reset_index( drop = True, inplace = True )

# Head.
dados_anos.head()


    #          #
    # Gráficos #
    #          # 
     
# Variação do loudness conforme os anos    
fig = px.line( dados_anos, x = 'year', y = 'loudness', markers = True, title = 'Variação do loudness conforme os anos' )
fig.show()

# Gerando gráfico de linhas com o go.
fig = go.Figure()

fig.add_trace( go.Scatter( x = dados_anos[ 'year' ], y = dados_anos[ 'acousticness' ],
                           name = 'Acousticness' ) )
fig.add_trace( go.Scatter( x = dados_anos[ 'year' ], y = dados_anos[ 'valence' ],
                           name = 'Valence' ) )
fig.add_trace( go.Scatter( x = dados_anos[ 'year' ], y = dados_anos[ 'danceability' ],
                           name = 'Danceability' ) )
fig.add_trace( go.Scatter( x = dados_anos[ 'year' ], y = dados_anos[ 'energy' ],
                           name = 'Energy' ) )
fig.add_trace( go.Scatter( x = dados_anos[ 'year' ], y = dados_anos[ 'instrumentalness' ],
                           name = 'Instrumentalness' ) )
fig.add_trace( go.Scatter( x = dados_anos[ 'year' ], y = dados_anos[ 'liveness' ],
                           name = 'Liveness' ) )
fig.add_trace( go.Scatter( x = dados_anos[ 'year' ], y = dados_anos[ 'speechiness' ],
                           name = 'Speechiness' ) )

fig.show()

# Correlações.
fig = px.imshow( dados.select_dtypes( include = np.number ).corr(), text_auto = True ) # Correlações mostrando o texto do que é cada variável.
fig.show()


    #                           #
    # Clusterização dos Gêneros #
    #                           # 
    
# Gêneros são únicos.
dados_generos[ 'genres' ].nunique()

# Dropando a coluna gênero.
dados_generos1 = dados_generos.drop( 'genres', axis = 1 )

# Fixando semente.
SEED = 1224
np.random.seed( SEED )

# Observação: 
# Pipeline é a forma na qual iremos gerir os dados dentro do processo de machine learning. 
# Aqui estamos padronizando e depois aplicando PCA com 2 componentes.
pca_pipeline = Pipeline( [ ( 'scaler', StandardScaler() ), ( 'PCA', PCA( n_components = 2, random_state = SEED ) ) ] )

# Criando a redução de dimensionalidade em 2 dimensões.
genre_embedding_pca = pca_pipeline.fit_transform( dados_generos1 )
projection = pd.DataFrame( columns = [ 'x', 'y' ], data = genre_embedding_pca )
projection

# KMeans.
kmeans_pca = KMeans( n_clusters = 5, verbose = True, random_state = SEED )
kmeans_pca.fit( projection )
dados_generos[ 'cluster_pca' ] = kmeans_pca.predict( projection )
projection[ 'cluster_pca' ]    = kmeans_pca.predict( projection )
projection

# Pegando gêneros.
projection[ 'generos' ] = dados_generos[ 'genres' ]
projection


    #                      #  
    # Plotando os Clusters #
    #                      #

fig = px.scatter(
     projection, x = 'x', y = 'y', color = 'cluster_pca', hover_data = [ 'x', 'y', 'generos' ]
) # hover_data é para dizer quais dados serão exibidos ao passar o mouse.
fig.show() 

# Olhando no PCA, do Pipeline, quanto da variância dos dados está sendo explicada.
pca_pipeline[ 1 ].explained_variance_ratio_.sum() # 0 é o scaler, 1 é o PCA.
# Quase 50% explicado.

# Quantas variáveis cada componente está explicando.
pca_pipeline[ 1 ].explained_variance_ # O primeiro componente explica quase 4 variáveis, já o segundo componente explica 1.5.


    #                           #
    # Clusterização das Músicas #
    #                           # 
    
# Frequência de artistas.
dados[ 'artists' ].value_counts()

# Frequência de músicas.
dados[ 'artists_song' ].value_counts() # São únicas.

# Definindo o OneHotEncoder.
ohe = OneHotEncoder( dtype = int ) # Prefiro inteiro ao booleano.
colunas_ohe = ohe.fit_transform( dados[ [ 'artists' ] ] ).toarray() # precisa estar em array.
dados2 = dados.drop( 'artists', axis = 1 ) # dropando coluna artista.

# Criando novos dados com os dados sem a coluna artista e as dummies do OneHotEncoder pegando o nome da feature (nome dos artistas).
dados_musicas_dummies = pd.concat( [ dados2, pd.DataFrame( colunas_ohe, columns = ohe.get_feature_names_out( [ 'artists' ] ) ) ], axis = 1 )
dados_musicas_dummies

# Criando PCA definindo taxa de aprendizagem de 70% ao invés do número de componentes.
pca_pipeline = Pipeline( [ ( 'scaler', StandardScaler() ), ( 'PCA', PCA( n_components = 0.7, random_state = SEED ) ) ] )

# Treinando Pipeline para os dados sem as variáveis de string.
music_embedding_pca = pca_pipeline.fit_transform( dados_musicas_dummies.drop( [ 'id', 'name', 'artists_song' ], axis = 1 ) )
projection_m = pd.DataFrame( data = music_embedding_pca )

# Número de componentes do PCA.
pca_pipeline[ 1 ].n_components_ # Número de componentes principais do PCA.

# Usando KMeans.
kmeans_pca_m = KMeans( n_clusters = 50, verbose = False, random_state = SEED )
kmeans_pca_m.fit( projection_m )
dados[ 'cluster_pca' ]        = kmeans_pca_m.predict( projection_m )
projection_m[ 'cluster_pca' ] = kmeans_pca_m.predict( projection_m )

# Trazendo informações de artista e nome da música.
projection_m[ 'artist' ] = dados[ 'artists' ]
projection_m[ 'song' ]   = dados[ 'artists_song' ]

# Exibindo.
projection_m


    #                      #  
    # Plotando os Clusters #
    #                      #

fig = px.scatter(
     projection_m, x = 0, y = 1, color = 'cluster_pca', hover_data = [ 0, 1, 'song' ]
) # hover_data é para dizer quais dados serão exibidos ao passar o mouse.
fig.show() # A coluna 0 e 1 são as que mais explicam o PCA.

# Olhando no PCA, do Pipeline, quanto da variância dos dados está sendo explicada.
pca_pipeline[ 1 ].explained_variance_ratio_.sum() # 0 é o scaler, 1 é o PCA. 70% explicado, como eu pedi.

# Quantas variáveis cada componente está explicando.
pca_pipeline[ 1 ].explained_variance_ 


    #                                                # 
    # Tentando montar o recomendador com os clusters #
    #                                                #
    
# Nome de uma música inicial.
nome_musica = 'Ed Sheeran - Shape of You' 

# Pegando o cluster desta música.
cluster = list( projection_m[ projection_m[ 'song' ] == nome_musica ][ 'cluster_pca' ] )[ 0 ]

# Valor x da música.
x_musica = list( projection_m[ projection_m[ 'song' ] == nome_musica ][ 0 ] )[ 0 ]

# Valor y da música.
y_musica = list( projection_m[ projection_m[ 'song' ] == nome_musica ][ 1 ] )[ 0 ]

# Possíveis músicas recomendadas do mesmo cluster.
musicas_recomendadas = projection_m[ projection_m[ 'cluster_pca' ] == cluster ][ [ 0, 1, 'song' ] ]

# Distâncias euclidianas da nossa música para cada música do mesmo cluster.
distancias = euclidean_distances( musicas_recomendadas[[ 0, 1 ]], [[ x_musica, y_musica ]] )

# Pegando o ID da música.
musicas_recomendadas[ 'id' ] = dados[ 'id' ]
musicas_recomendadas[ 'distancias' ] = distancias # adicionando as distâncias.
recomendada = musicas_recomendadas.sort_values( 'distancias' ).head( 10 ) # 10 músicas mais recomendadas.
recomendada