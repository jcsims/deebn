(defproject deebn "0.1.0"
  :description "Deep Belief Network using Restricted Boltzmann Machines"
  :url "https://jcsi.ms"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.6.0"]
                 [org.clojure/tools.reader "0.8.9"]
                 [org.clojure/data.csv "0.1.2"]
                 [net.mikera/vectorz-clj "0.26.1"]
                 [net.mikera/core.matrix "0.30.2"]
                 [net.mikera/core.matrix.stats "0.4.1"]
                 [com.taoensso/timbre "3.3.1"]]
  :profiles {:dev {:dependencies [[org.clojure/test.check "0.5.9"]]}}
  :jvm-opts ["-Xmx4g" "-Xms3g"]
  :deploy-repositories [["releases" :clojars]])
