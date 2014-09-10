(defproject deebn "0.1.0-SNAPSHOT"
  :description "Deep Belief Network using Restricted Boltzmann Machines"
  :url "https://jcsi.ms"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.6.0"]
                 [org.clojure/tools.reader "0.8.8"]
                 [org.clojure/data.csv "0.1.2"]
                 [net.mikera/vectorz-clj "0.25.0"]
                 [net.mikera/core.matrix "0.29.1"]
                 [com.taoensso/timbre "3.3.1"]]
  :profiles {:dev {:dependencies [[org.clojure/test.check "0.5.9"]]}}
  :jvm-opts ["-Xmx4g" "-Xms3g"])
