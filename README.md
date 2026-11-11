# Importa a biblioteca pandas, que é ótima para trabalhar com dados em tabelas (como planilhas).
# A gente costuma dar um "apelido" para ela, que é 'pd'.
from pathlib import Path
import pandas as pd

# Importa algumas coisas da biblioteca 'rich' para deixar o terminal mais bonito e organizado.
try:
    from rich import print
    from rich.console import Console
    from rich.panel import Panel
except ImportError:
    from builtins import print

    class Console:
        def print(self, value):
            print(value)

    class Panel:
        def __init__(self, value, **kwargs):
            self.value = value

        def __str__(self):
            return self.value

# Importa funções da biblioteca 'scikit-learn', que é a principal para fazer Machine Learning em Python.
from sklearn.metrics import accuracy_score  # Para calcular a acurácia (o quão bem o modelo acerta).
from sklearn.model_selection import train_test_split  # Para dividir nossos dados em treino e teste.
try:
    from sklearn.tree import plot_tree  # Para desenhar a árvore de decisão.
    import matplotlib.pyplot as plt  # Para mostrar gráficos, como a árvore de decisão.
except ImportError:
    plot_tree = None
    plt = None
from sklearn.tree import DecisionTreeClassifier  # O modelo da Árvore de Decisão.
from sklearn import svm  # Onde fica o modelo SVM (Support Vector Machine).

# Carregando os dados de um arquivo CSV para uma tabela do pandas chamada 'df' (DataFrame).
# O arquivo precisa estar no seu Google Drive, nesse caminho.
BASE_DIR = Path(__file__).resolve().parent
input_path = BASE_DIR / 'dados_loja_roupas.csv'
df = pd.read_csv(input_path)

# Cria um 'Console' da biblioteca rich para podermos usar os painéis bonitinhos.
caixa = Console()

# --- Variáveis Globais ---
# São como caixinhas que guardam informações e podem ser usadas em qualquer parte do código.
# A gente começa elas como 'None' (vazio) porque elas só vão ter valor depois que a função 'preparar_dados' for chamada.

clf = None  # Vai guardar nosso modelo de Árvore de Decisão treinado.
svm_model = None  # Vai guardar nosso modelo SVM treinado.
X_train, X_test, y_train, y_test = None, None, None, None # Vão guardar nossos dados de treino e teste.

# --- Função para preparar os dados e treinar os modelos ---
def preparar_dados():
    # A palavra 'global' avisa que vamos usar e modificar as variáveis globais que criamos lá fora.
    global X_train, X_test, y_train, y_test, clf, svm_model

    # --- Definir variáveis ---
    # 'X' (com letra maiúscula) vai ser a nossa tabela só com as colunas que usamos para prever (as características do cliente).
    X = df[['Gasto_Mes_Anterior', 'Visitas_Loja_Mes', 'Tempo_Medio_Loja_Minutos',
            'Compra_Promocao', 'Idade_Cliente', 'Itens_Comprados', 'Usa_Fidelidade']]
    # 'y' (com letra minúscula) é o que a gente quer prever: a coluna 'Tipo_Cliente'.
    y = df['Tipo_Cliente']

    # --- Dividir os dados ---
    # Essa função mágica divide nossos dados em dois grupos:
    # 1. Grupo de Treino: para ensinar o modelo (70% dos dados).
    # 2. Grupo de Teste: para ver se o modelo aprendeu direito (30% dos dados).
    # 'random_state=1' garante que a divisão seja sempre a mesma, para que os resultados sejam consistentes.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # --- Treinar modelos ---
    # Cria o modelo de Árvore de Decisão. 'random_state=1' ajuda a ter um resultado consistente.
    clf = DecisionTreeClassifier(random_state=1)
    # Ensina o modelo de árvore usando os dados de treino. Isso é o "treinamento".
    clf.fit(X_train, y_train)

    # Cria o modelo SVM. O 'kernel="linear"' é um tipo de configuração para esse modelo.
    svm_model = svm.SVC(kernel='linear')
    # Ensina o modelo SVM com os mesmos dados de treino.
    svm_model.fit(X_train, y_train)

    # --- Adicionar as previsões ao DataFrame original ---
    # Usa o modelo de árvore treinado para classificar TODOS os clientes do DataFrame original.
    df['Classificacao_Arvore'] = clf.predict(X)
    # Faz o mesmo com o modelo SVM.
    df['Classificacao_SVM'] = svm_model.predict(X)

    # --- Salvar o novo arquivo CSV ---
    # Define o caminho e o nome do novo arquivo que será salvo no Google Drive.
    output_path = BASE_DIR / "dados_loja_roupas_com_classificacoes.csv"
    # Salva o DataFrame (que agora tem as duas novas colunas de classificação) em um novo arquivo CSV.
    # 'index=False' evita que o pandas salve uma coluna extra com os números das linhas.
    df.to_csv(output_path, index=False)


# --- Função do Menu Principal ---
def menu_principal():
    # Primeiro, chama a função para preparar tudo: carregar dados, dividir e treinar os modelos.
    preparar_dados()
    # 'while True' cria um loop infinito, o menu vai ficar aparecendo até a gente escolher sair.
    while True:
        # Usa o 'Panel' do rich para criar uma caixa de título bonita no terminal.
        caixa.print(Panel("[bold]      MENU PRINCIPAL   [/bold]", width=30, padding=(0, 0)))
        # Mostra as opções para o usuário.
        print("1 - Árvore de Decisão")
        print("2 - SVM")
        print("3 - Terminar")
        # Pede para o usuário digitar a escolha dele.
        opcao = input("Escolha uma opção: ")

        # Verifica qual foi a escolha do usuário.
        if opcao == "1":
            # Se escolheu 1, chama a função do menu da árvore de decisão.
            menu_arvore_decisao()
        elif opcao == "2":
            # Se escolheu 2, chama a função do menu do SVM.
            menu_svm()
        elif opcao == "3":
            # Se escolheu 3, avisa que o programa está encerrando.
            print("Programa encerrado.")
            # 'break' é o comando que quebra o loop infinito e encerra o programa.
            break
        else:
            # Se digitou qualquer outra coisa, avisa que a opção é inválida.
            print("Opção inválida.")

# --- Função do Menu da Árvore de Decisão ---
def menu_arvore_decisao():
    # Outro loop infinito, para o submenu da árvore.
    while True:
        # Mostra o título do submenu.
        caixa.print(Panel("[bold]      ÁRVORE DE DECISÃO   [/bold]", width=30, padding=(0, 0)))
        print("1 - Mostrar Desempenho")
        print("2 - Mostrar Árvore")
        print("3 - Fazer Nova Classificação")
        print("4 - Voltar")
        opcao = input("Escolha uma opção: ")

        if opcao == "1":
            # Usa o modelo treinado ('clf') para prever os resultados dos dados de teste.
            y_pred = clf.predict(X_test)
            # Compara as previsões ('y_pred') com os resultados reais ('y_test') para ver a acurácia.
            accuracy = accuracy_score(y_test, y_pred)
            # Mostra a acurácia em porcentagem, formatada com 2 casas decimais, dentro de um painel verde.
            caixa.print(Panel(f"[green]Acurácia: {accuracy*100:.2f}%[/green]"))

        elif opcao == "2":
            if plt is None or plot_tree is None:
                caixa.print("Para mostrar a árvore, instale matplotlib: python -m pip install matplotlib")
                continue
            # Cria uma "tela" para o nosso gráfico com um tamanho bom (15 de largura por 10 de altura).
            plt.figure(figsize=(15,10))
            # A função que desenha a árvore!
            plot_tree(clf,  # o modelo treinado
                      filled=True,  # preenche os nós com cores
                      feature_names=X_train.columns,  # nomes das características (colunas)
                      class_names=clf.classes_,  # nomes das classes (ex: 'Ouro', 'Prata')
                      rounded=True)  # deixa os cantos dos nós arredondados
            # Mostra o gráfico da árvore na tela.
            plt.show()

        elif opcao == "3":
            # Aqui, pedimos ao usuário para digitar os dados de um novo cliente.
            # Cada 'input' mostra uma mensagem e espera o usuário digitar algo.
            # 'float()' e 'int()' transformam o texto digitado em número (decimal ou inteiro).
            novo_cliente = {
                'Gasto_Mes_Anterior': float(input("Gasto no mês anterior: ")),
                'Visitas_Loja_Mes': int(input("Visitas à loja no último mês: ")),
                'Tempo_Medio_Loja_Minutos': float(input("Tempo médio na loja (min): ")),
                'Compra_Promocao': int(input("Comprou em promoção? (1-Sim/0-Não): ")),
                'Idade_Cliente': int(input("Idade do cliente: ")),
                'Itens_Comprados': int(input("Itens comprados: ")),
                'Usa_Fidelidade': int(input("Usa fidelidade? (1-Sim/0-Não): "))
            }
            # Cria um novo DataFrame do pandas com os dados do novo cliente.
            # O modelo espera receber os dados nesse formato.
            novo_df = pd.DataFrame([novo_cliente])
            # Usa o modelo de árvore para prever a classificação do novo cliente.
            predicao = clf.predict(novo_df)
            # Mostra a previsão na tela, dentro de um painel azul.
            # 'predicao[0]' pega o primeiro (e único) item da lista de previsões.
            caixa.print(Panel(f"[blue]Classificação: {predicao[0]}[/blue]"))

        elif opcao == "4":
            # Se escolher 4, 'break' quebra o loop deste submenu e volta para o menu principal.
            break
        else:
            print("Opção inválida.")

# --- Função do Menu do SVM ---
def menu_svm():
    # Loop infinito para o submenu do SVM.
    while True:
        # Mostra o título e as opções.
        caixa.print(Panel("[bold]           SVM            [/bold]", width=30, padding=(0, 0)))
        print("1 - Mostrar Desempenho")
        print("2 - Fazer Nova Classificação")
        print("3 - Voltar")
        opcao = input("Escolha uma opção: ")

        if opcao == "1":
            # Usa o modelo SVM para prever os resultados dos dados de teste.
            y_pred = svm_model.predict(X_test)
            # Calcula a acurácia comparando previsões e resultados reais.
            accuracy = accuracy_score(y_test, y_pred)
            # Mostra a acurácia formatada.
            caixa.print(Panel(f"[green]Acurácia: {accuracy*100:.2f}%[/green]"))

        elif opcao == "2":
            # Esta parte é igual à da árvore de decisão: pede os dados do novo cliente.
            novo_cliente = {
                'Gasto_Mes_Anterior': float(input("Gasto no mês anterior: ")),
                'Visitas_Loja_Mes': int(input("Visitas à loja no último mês: ")),
                'Tempo_Medio_Loja_Minutos': float(input("Tempo médio na loja (min): ")),
                'Compra_Promocao': int(input("Comprou em promoção? (1-Sim/0-Não): ")),
                'Idade_Cliente': int(input("Idade do cliente: ")),
                'Itens_Comprados': int(input("Itens comprados: ")),
                'Usa_Fidelidade': int(input("Usa fidelidade? (1-Sim/0-Não): "))
            }
            # Cria o DataFrame para o novo cliente.
            novo_df = pd.DataFrame([novo_cliente])
            # Usa o modelo SVM ('svm_model') para fazer a previsão.
            predicao = svm_model.predict(novo_df)
            # Mostra o resultado.
            caixa.print(Panel(f"[blue]Classificação: {predicao[0]}[/blue]"))

        elif opcao == "3":
            # Quebra o loop e volta para o menu principal.
            break
        else:
            print("Opção inválida.")

# --- Iniciar programa ---
# Depois de definir todas as funções, esta é a linha que realmente começa o programa,
# chamando a função do menu principal.
menu_principal()


