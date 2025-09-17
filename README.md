## Contexto
Códigos utilizados para o projeto de Iniciação Científica "**Impacto da Propagação de Erro na Classificação Sequencial de Posicionamento em Threads de Redes Sociais**".

Universidade de São Paulo - Orientador: Prof. Dr. Luciano Antonio Digiampietri.

## Resumo
A detecção de posicionamento no Processamento de Língua Natural (PLN) tem ganhado relevância devido ao crescente volume de interações em plataformas digitais, onde a análise automática de debates pode auxiliar no monitoramento de opiniões e na identificação de desinformação. Entretanto, os métodos tradicionais frequentemente se concentram em postagens isoladas, ignorando que, em discussões online, o contexto conversacional e a estrutura hierárquica das threads exercem um papel fundamental para a correta interpretação. Muitos dos modelos que consideram essa estrutura interdependente se apoiam em predições anteriores, tornando-se suscetíveis à propagação de erro, onde uma falha inicial compromete a classificação de toda a sequência subsequente.

Dessa forma, o intuito deste trabalho é investigar o impacto do fenômeno de propagação de erro na classificação sequencial de posicionamento. Para isso, o processo utilizado foi da comparação do desempenho entre modelos "independentes", que analisam cada comentário de forma isolada, e modelos "dependentes", que utilizam o rótulo previsto do comentário anterior ("pai") como uma variável preditora para explorar explicitamente o impacto do erro. 

Para avaliar o desempenho dos modelos, foram utilizadas as métricas de acurácia, F1-Macro e F1-Weighted, com análises agregadas e por grau de profundidade na thread. Os resultados obtidos mostraram que os modelos que utilizam o contexto do comentário pai (seja seu texto ou seu rótulo real) alcançam resultados ligeiramente superiores em relação ao modelo sem contexto. No entanto, a abordagem sequencial, que espelha o uso em produção, teve seu desempenho degradado, com a acurácia caindo de 50% para 44%, o que confirma a hipótese da propagação de erro. Essa tendência de queda se mostrou mais acentuada em níveis mais profundos da conversa, evidenciando que erros nos comentários iniciais se acumulam e comprometem as predições subsequentes.

Etapas principais do fluxo de scripts:
1. Carrega arquivo de dados pré-processados (features prontas).
2. Separa treino e teste por thread (para evitar vazamento de informação).
3. Treina um modelo dependente (usa emb + target_emb + parent_label).
4. Gera, no teste, as labels de pais de forma sequencial por profundidade.
5. Avalia desempenho com parent_label real (base) e com parent_label
   previsto (sequencial), incluindo métricas por grau de profundidade.
