import Dependencies._

ThisBuild / scalaVersion     := "2.11.12"
ThisBuild / version          := "0.1.0-SNAPSHOT"
ThisBuild / organization     := "com.example"
ThisBuild / organizationName := "example"

fork := true

lazy val root = (project in file("."))
  .settings(
    name := "deepLearningEx",
    libraryDependencies += munit % Test,
    // Added for HelloWorld (possibly shared by other examples)
    // All DeepLearning.scala built-in plugins.
    libraryDependencies += "com.thoughtworks.deeplearning" %% "plugins-builtins" % "2.0.0",

    // The native backend for nd4j.
    libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.8.0",
    libraryDependencies += "org.nd4j" %% "nd4s" % "0.8.0",

    // Uncomment the following line to switch to the CUDA backend for nd4j.
    // libraryDependencies += "org.nd4j" % "nd4j-cuda-8.0-platform" % "0.8.0"

    // The magic import compiler plugin, which may be used to import DeepLearning.scala distributed in source format.
    addCompilerPlugin("com.thoughtworks.import" %% "import" % "2.0.0"),

    // The ThoughtWorks Each library, which provides the `monadic`/`each` syntax.
    libraryDependencies += "com.thoughtworks.each" %% "each" % "3.3.1",

    addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full),

    // Needed for image classification
    libraryDependencies += "com.chuusai" %% "shapeless" % "2.3.2",
  )

// See https://www.scala-sbt.org/1.x/docs/Using-Sonatype.html for instructions on how to publish to Sonatype.
