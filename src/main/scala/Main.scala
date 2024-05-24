import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, VectorAssembler, Word2Vec, Word2VecModel}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.plot.CategoryPlot
import org.jfree.chart.renderer.category.BarRenderer
import org.jfree.data.category.DefaultCategoryDataset
import javax.swing.{JFrame, JScrollPane, JTable}
import java.awt.{Color, BorderLayout}
import javax.swing.table.DefaultTableModel

object Main {
  def main(args: Array[String]): Unit = {
    // Initialisation de SparkSession
    val spark = SparkSession.builder()
      .appName("CustomerCommentsAnalysis")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._ // Importer les conversions implicites pour permettre l'utilisation de $

    // Chargement des données des commentaires
    val commentsDF = spark.read.option("header", "true").csv("comments.csv")

    // Nettoyage des données et tokenization
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("comment_text")
      .setOutputCol("words")
      .setPattern("\\W") // Utilisation des caractères non-alphanumériques comme séparateurs

    val tokenizedDF = regexTokenizer.transform(commentsDF)

    // Suppression des mots vides (stop words)
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered_words")

    val cleanedDF = stopWordsRemover.transform(tokenizedDF)

    // Entraînement du modèle Word2Vec pour obtenir des représentations vectorielles des mots
    val word2Vec = new Word2Vec()
      .setInputCol("filtered_words")
      .setOutputCol("new_word_vectors") // Renommer la colonne pour éviter les conflits
      .setVectorSize(100) // Taille des vecteurs
      .setMinCount(5) // Nombre minimum d'occurrences pour un mot

    val word2VecModel = word2Vec.fit(cleanedDF)
    val vectorizedDF = word2VecModel.transform(cleanedDF).drop("word_vectors") // Supprimer la colonne existante

    // Assembler les colonnes de features
    val assembler = new VectorAssembler()
      .setInputCols(Array("new_word_vectors")) // Colonnes contenant les vecteurs de mots
      .setOutputCol("features")

    val assembledDF = assembler.transform(vectorizedDF)

    // Clustering des commentaires pour identifier les thèmes récurrents
    val kMeans = new KMeans()
      .setK(5) // Nombre de clusters à identifier
      .setSeed(123) // Graine pour la reproductibilité
      .setFeaturesCol("features") // Spécifier la colonne de features

    val pipeline = new Pipeline().setStages(Array(kMeans))
    val clusteredModel = pipeline.fit(assembledDF)

    val clusteredDF = clusteredModel.transform(assembledDF)

    // Classer les commentaires en bons et mauvais
    val groupedDF = clusteredDF.groupBy("prediction").agg(collect_list("comment_text").alias("comments"))

    // Convert data to Array[Array[Object]]
    val rowData: Array[Array[Object]] = groupedDF.collect().map(row => Array(row.getAs[Int]("prediction").asInstanceOf[Object], row.getAs[Seq[String]]("comments").mkString(", ").asInstanceOf[Object]))
    // Classer les commentaires en bons et mauvais
    val goodCommentsDF = clusteredDF.filter($"prediction" === 0) // Par exemple, supposons que le cluster 0 représente les bons commentaires
    val badCommentsDF = clusteredDF.filter($"prediction" === 1) // Par exemple, supposons que le cluster 1 représente les mauvais commentaires

    // Compter le nombre de bons et de mauvais commentaires
    val goodCount = goodCommentsDF.count().toDouble
    val badCount = badCommentsDF.count().toDouble
    // Convert column names to Array[Object]
    val columnNames: Array[Object] = Array("Prediction", "Comments").asInstanceOf[Array[Object]]

    // Créer le dataset pour JFreeChart
    val dataset = new DefaultCategoryDataset()
    dataset.addValue(goodCount, "Comments", "Good") // Assuming rowData contains good comments
    dataset.addValue(badCount, "Comments", "Bad") // Assuming rowData does not contain bad comments

    // Créer le graphique à barres avec JFreeChart
    val chart = ChartFactory.createBarChart(
      "Good vs Bad Comments", // Titre du graphique
      "Category", // Axe des x
      "Count", // Axe des y
      dataset // Dataset
    )

    // Personnaliser les couleurs des barres
    val plot = chart.getPlot().asInstanceOf[CategoryPlot]
    val renderer = plot.getRenderer().asInstanceOf[BarRenderer]
    renderer.setSeriesPaint(0, Color.GREEN) // Série pour les bons commentaires en vert
    renderer.setSeriesPaint(1, Color.RED) // Série pour les mauvais commentaires en rouge

    // Créer le modèle de table avec les données
    val tableModel = new DefaultTableModel(rowData.map(_.asInstanceOf[Array[AnyRef]]), columnNames)

    // Créer la table avec le modèle de table
    val table = new JTable(tableModel)

    // Set preferred column widths based on content
    for (columnIndex <- 0 until table.getColumnCount
         ) {
      var maxWidth = 0
      // Iterate over rows to find the maximum width of content in the column
      for (rowIndex <- 0 until table.getRowCount) {
        val renderer = table.getCellRenderer(rowIndex, columnIndex)
        val component = table.prepareRenderer(renderer, rowIndex, columnIndex)
        val width = component.getPreferredSize.width
        maxWidth = math.max(maxWidth, width)
      }
      // Set the preferred width of the column
      table.getColumnModel.getColumn(columnIndex).setPreferredWidth(maxWidth)
    }

    // Create a frame for the table
    val tableFrame = new JFrame("Comments Table")
    tableFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
    tableFrame.add(new JScrollPane(table))
    tableFrame.pack()
    tableFrame.setVisible(true)

    // Create a frame for the chart
    val chartFrame = new JFrame("Comments Chart")
    chartFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
    chartFrame.add(new ChartPanel(chart))
    chartFrame.pack()
    chartFrame.setVisible(true)

    // Fermeture de la session Spark
    spark.stop()
  }
}
