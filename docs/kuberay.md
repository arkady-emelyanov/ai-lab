[← README](../README.md) · **KubeRay**

# KubeRay

## Overview

In k3s mode the lab runs [KubeRay](https://github.com/kuberay-project/kuberay), the Kubernetes operator for Ray, with its API server, its web dashboard and its History Server (`kuberay: true`, the default; `kuberay: false` in `local.yml` leaves it out; turning it off on a built lab takes `make down` and `make up`, [as for `scheduler`](platform.md#configuration)). Ray clusters, jobs and services are Kubernetes resources (`RayCluster`, `RayJob`, `RayService`); KubeRay starts their head and worker pods on the trays, Kueue admits them against the GPU quota, and the dashboard lists them in the browser, finished ones included: their logs and events are kept in the lab's object storage (RustFS).

KubeRay is installed from its release (1.7.1, `kuberay_version`): the operator and API server charts from `kuberay-project.github.io/kuberay-helm` and the images from `quay.io/kuberay`, the History Server's and the collector's included (the History Server is beta upstream). Ray's dashboard for finished clusters is Ray's own image, `docker.io/rayproject/ray` (`kuberay_history_ray_image`), of the venv's Ray version (`ray_version`, 2.59.0).

| Component | Where | Role |
|---|---|---|
| KubeRay operator (chart `kuberay-operator`) | namespace `kuberay`, on `sched-control` | reconciles `RayCluster`, `RayJob`, `RayService`, `RayCronJob`: head and worker pods, their services, job submission |
| KubeRay API server V2 (chart `kuberay-apiserver`) | namespace `kuberay`, on `sched-control` | Kubernetes' own API for Ray's resources and their events, and the Ray dashboards of KubeRay's services, for the dashboard |
| KubeRay dashboard (`quay.io/kuberay/dashboard`) | namespace `kuberay`, on `sched-control` | web UI: Ray jobs and clusters per namespace, their state, creating and deleting them, links to each cluster's Ray dashboard |
| KubeRay History Server (`quay.io/kuberay/historyserver`) | namespace `kuberay`, on `sched-control` | serves finished Ray clusters (nodes, jobs, tasks, actors, logs) from the RustFS bucket `ray-history`, for the dashboard's **History** pages |
| Ray's dashboard for finished clusters (`rayproject/ray:2.59.0-py312-cpu`) | namespace `kuberay`, on `sched-control` | `http://ray-history.<controller address>.nip.io`: Ray's own dashboard with the History Server as its backend (proxy mode): a finished cluster's jobs, tasks, actors and logs as on a live one; where a finished job's **Logs** link lands |
| Prometheus job `ray`, Grafana folder **Ray** | the controller's Prometheus and Grafana | Ray's metrics from every Ray node in KubeRay's pods, and Ray's own Grafana dashboards, which Ray's dashboard embeds panels of |
| Collector (`quay.io/kuberay/collector`) | a sidecar in each Ray pod that asks for it (`ray-job.yaml` does) | uploads the node's Ray logs and events to `ray-history` while the cluster runs and as it shuts down |
| S3 user `ray-history`, Secret `ray-history-s3`; with Run:ai, the credentials asset `ray-history-s3` and Secret `genericsecret-ray-history-s3` | RustFS; namespaces `kuberay` and every user's; with Run:ai, every `runai-<project>` | the History Server's bucket and its keys (`.secrets/ray-history.s3`), allowed on that bucket only |
| ConfigMap `ray-grafana` | every user's namespace; with Run:ai, also namespace `runai`, labelled `run.ai/cluster-wide` (Run:ai copies it to every project's) | where a Ray head finds Prometheus and Grafana for the Grafana panels in its Ray dashboard |
| `MutatingAdmissionPolicy` `rayjob-shutdown-after-finish` | cluster-wide | sets `shutdownAfterJobFinishes` on RayJobs created without it (the dashboard's job form), which Kueue requires |
| Ingresses `kuberay`, `kuberay-historyserver`, `ray-history`, `ray-history-open` | HAProxy ingress (k3s add-on), `sched-control` port 80 | `http://kuberay.<controller address>.nip.io`: the API server's paths (`/apis/ray.io/`, `/apis/events.k8s.io/`, `/api/v1/`) to the API server, `/historyserver/` to the History Server, everything else to the dashboard; `http://ray-history.<controller address>.nip.io`: Ray's dashboard, `/open/` to the History Server |

The KubeRay services use about 100 MiB on the controller and Ray's dashboard for finished clusters about 630 MiB (Ray's head processes; its dashboard modules are idle in proxy mode but cannot be left out); the ingress controller uses about 190 MiB. Ray's own pods run on the trays.

## How it works

### Installation

The `kuberay` role (`roles/kuberay`, run by `make configure` after the k3s cluster configuration) writes one k3s manifest (`lab-kuberay.yaml` in k3s' manifests directory): the two charts as `HelmChart` resources, like the lab's other add-ons, the dashboard and the History Server, which have no charts upstream, as Deployments and Services, the ingresses, the admission policy and the History Server's storage Secrets. All five services (the operator, the API server, the dashboard, the History Server and Ray's dashboard for finished clusters, Deployment `ray-history`) run on the control-plane node, with small memory requests (KubeRay's charts request 512 and 300 MiB, for 30 and 10 MiB used). Kueue starts managing Ray's resources by itself once their APIs exist. The role runs after the storage node's, which creates the S3 user and the bucket (`roles/s3users`). With Run:ai it also runs after the `runai` role, and creates the credentials asset `ray-history-s3` through Run:ai's API (replacing it when its keys change: Run:ai's API cannot edit them).

### Dashboard and API server

The dashboard is a web application that the browser runs: it reads where the API server is from the dashboard's `/api/config` (`API_DOMAIN`), then calls the API server directly. Both are behind the same name, so the browser talks to one origin and no CORS is needed. The API server V2 passes requests for `ray.io` resources to Kubernetes with its own service account, and proxies `/api/v1/namespaces/<ns>/services/<service>/proxy/` for services KubeRay created, which is how the dashboard links a cluster's Ray dashboard (port 8265 of its head service).

Everything is plain HTTP: the lab has no certificates to import for it. HTTPS stays only where a component requires it (Run:ai, [NVIDIA Run:ai](runai.md)).

The dashboard opens on namespace `joe` (the first directory user); its namespace selector switches to others.

### History

A running cluster's Ray dashboard dies with the cluster, and so does the **Logs** link of its job on the **Jobs** page ("kuberay service not found"). KubeRay's History Server keeps them. A collector runs next to Ray in each pod that has one, reads Ray's logs from the directory it shares with Ray (`/tmp/ray`, an `emptyDir`) and receives Ray's events (tasks, actors, jobs, nodes; Ray 2.55 or later, the venv has 2.59), and uploads both to the bucket `ray-history`, under the cluster's name and Ray session. The collector uploads what is left when its pod stops, so a RayJob can delete its cluster right after the job: `ray-job.yaml` waits 5 s (`ttlSecondsAfterFinished`). Upstream's samples wait 90 s, but Kueue gives a job's quota back when the job ends, and a cluster kept that long holds its GPUs while Kueue admits the next workload onto them, which then waits (`Insufficient nvidia.com/gpu`).

The dashboard's **History** pages list the finished clusters, and for one of them its nodes, tasks and every log file, the driver's included (`job-driver-<id>.log`). The dashboard's server asks the History Server through `/historyserver/` on the dashboard's name, where HAProxy fixes one path: KubeRay 1.7.1's dashboard opens a cluster at `/enter_cluster/<namespace>/<cluster>/<session>`, its History Server expects `/enter_cluster/<namespace>/raycluster/<cluster>/<session>` (still so on KubeRay's main branch).

The History Server serves finished clusters only (proxying to live ones is off, upstream's default since 1.7.1) and keeps a few parsed sessions in memory (`--session-cache-max-memory=256Mi`).

**Ray's dashboard for finished clusters.** Upstream's way to look at a finished cluster is Ray's own dashboard with the History Server as its backend (`ray start --proxy-server-url=<History Server>`, Ray 2.55 or later): it sends every request to the History Server and serves its own pages where the History Server has none, and a cookie the History Server sets selects the cluster. The lab runs it at `http://ray-history.<controller address>.nip.io` (`kuberay_history_domain`, image `kuberay_history_ray_image`); without a cluster selected it shows the History Server's list of clusters (`/select_cluster`). `/open/<namespace>/<cluster>` selects a cluster's latest session and opens its **Jobs** page (HAProxy turns the History Server's `/enter_cluster` answer into a redirect).

A job's **Logs** link on the KubeRay dashboard's Jobs page is its cluster's live Ray dashboard (`/api/v1/namespaces/<ns>/services/<cluster>-head-svc:8265/proxy/#/jobs/<submission id>`), hard-coded in KubeRay 1.7.1's dashboard. Once the cluster is gone the API server answers 404, and HAProxy redirects the browser to `/open/<namespace>/<cluster>` on Ray's dashboard for finished clusters: the click lands on that cluster's Jobs page, and the job there opens with its logs. It cannot open the job itself: the History Server finds a job by its Ray job ID only, not by the submission ID in the link (also so on KubeRay's main branch).

### Scheduling

Kueue manages `RayJob` and `RayCluster` (its Ray integrations are on by default): a Ray cluster waits suspended until Kueue admits all of its pods at once against the ClusterQueue `gpu`, like a JobSet ([Kubernetes](kubernetes.md)). The examples ask for queue `gpu`; a Ray resource without a queue label in a user namespace goes to its `default` queue. Kueue requires `shutdownAfterJobFinishes: true` on a RayJob, so its cluster's resources return to the queue when the job ends; a `MutatingAdmissionPolicy` (`rayjob-shutdown-after-finish`) sets it on RayJobs created without it, such as the dashboard's. Pod templates can carry Kueue's topology annotations (`kueue.x-k8s.io/podset-required-topology: accelerator.topograph.run/domain`) to keep a Ray cluster in one NVLink domain.

### Ray in the pods

KubeRay generates each Ray container's `ray start` command (`--block`, the head's dashboard on all addresses, the GPUs from the container's `nvidia.com/gpu` limit) and runs it with `bash -c`. The lab's examples use the image the other Kubernetes examples use (`buildpack-deps:noble`) with the shared venv on `/shared` first in `PATH`, so Ray, PyTorch and the fake GPU libraries are the same as for every other job (`make frameworks`). Images with Ray inside (`rayproject/ray`) work too.

### Metrics

The lab's Prometheus (on the controller, outside Kubernetes) scrapes every Ray node's metrics port (KubeRay's `metrics`, 8080) in any namespace: job `ray`, with the labels Ray's dashboards expect (`ray_io_cluster`, plus `ray_io_node_type`, `namespace`, `pod`). Ubuntu's Prometheus package has no Kubernetes discovery, so `ray-targets` on the controller writes the running Ray pods, without those being deleted, to `/etc/prometheus/ray-targets.json` (Prometheus' `file_sd`) every 15 s, from `kubectl`. Ray's own Grafana dashboards (default, Data, Train, Serve, Serve LLM) are in the folder **Ray**, taken from what the lab's Ray generates (the Ray of `ray-history`, the venv's version), so they match its metrics.

A Ray head shows panels of those dashboards in its Ray dashboard (Overview, Metrics) when it knows where Prometheus and Grafana are: the ConfigMap `ray-grafana` in every user namespace has Ray's settings (`RAY_PROMETHEUS_HOST`, `RAY_GRAFANA_HOST` by the controller's name, `RAY_GRAFANA_IFRAME_HOST` by its address for the browser), and `ray-job.yaml` loads it into its Ray containers (`envFrom`). Grafana lets them be embedded and shown without a login: with KubeRay it allows embedding and anonymous viewers (read-only, `GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer`), as Ray documents. Finished clusters' metrics stay in Prometheus (`SessionName`, `Cluster` in Ray's dashboards); Ray's dashboard for finished clusters shows no panels, since the History Server does not answer Ray's Grafana checks for past sessions (501, "not yet supported").

### With Run:ai

With `runai: true`, a Ray cluster, job or service created in a Run:ai project's namespace (`runai-<project>`) is a Run:ai workload: Run:ai knows KubeRay's resources (its workload types `RayCluster`, `RayJob`, `RayService`), its pod grouper puts all of a Ray cluster's pods in one pod group, and KAI (`runai-scheduler`) schedules them together against the project's quota, as any Run:ai workload; the UI and API list it (type `RayCluster`, its GPUs). Nothing changes in KubeRay for it (no `batchScheduler`). Kueue leaves those namespaces alone (no LocalQueue there); in the users' namespaces Ray stays with Kueue.

What Ray clusters use in user namespaces reaches the projects' namespaces through Run:ai: the ConfigMap `ray-grafana` (labelled `run.ai/cluster-wide` in namespace `runai`, which Run:ai copies to every project), and the History Server's keys as a cluster-wide Run:ai credentials asset `ray-history-s3`, which Run:ai turns into the Secret `genericsecret-ray-history-s3` in each project's namespace (it adds the prefix). A Ray cluster in a project loads that Secret where `ray-job.yaml` loads `ray-history-s3`, and leaves out Kueue's queue label. Directory users such as `joe` have Kubernetes rights in their own namespace only: Ray clusters in project namespaces are created by the cluster admin (`bin/kubectl`).

## Usage

From the login node (`bin/ssh login`) as a directory user:

```
cd examples/kubernetes
./submit --wait ray-job.yaml           # RayJob: prints its name, writes the driver's output to <name>.out
kubectl get rayjobs,rayclusters,workloads
```

`ray-job.yaml` is the Ray example (`ray_demo.py`, one GPU task per GPU) as a `RayJob`: a head on one tray and a worker on the other, 4 GPUs each. KubeRay starts the cluster when Kueue admits it, submits the driver from a submitter pod (Job `<name>`, whose log is the driver's output), and deletes the cluster 5 s after the job ends. Its collectors keep the cluster's logs and events: the dashboard's **History** shows them after that.

Open `http://kuberay.10.107.111.10.nip.io` (your bridge's prefix may differ): **Jobs** and **Clusters** list Ray's resources in the selected namespace, with their state; a running cluster links its Ray dashboard (tasks, actors, logs). **History** lists finished clusters: open one, then its nodes' logs, tasks and the driver's output. A finished job's **Logs** link opens its cluster in Ray's own dashboard at `http://ray-history.10.107.111.10.nip.io` (Jobs, then the job: its driver logs; also tasks, actors, nodes), which also lists every finished cluster by itself.

For Grafana panels in your own Ray clusters' dashboards, load the ConfigMap `ray-grafana` into the Ray containers (`envFrom`, as `ray-job.yaml` does); their metrics reach Prometheus either way. To keep the history of your own Ray clusters, add the collector next to each Ray container as `ray-job.yaml` does: the container `collector`, Ray's event variables (`RAY_enable_ray_event` and the three after it), and the shared `/tmp/ray` volume. The keys come from the Secret `ray-history-s3` in your namespace. Users can also create and delete Ray jobs and clusters there; the dashboard's job form fills in `rayproject/ray` as the image (several GB to pull the first time) and no queue label, so Kueue puts the job in the namespace's `default` queue.

As cluster admin from your machine: `bin/kubectl get rayclusters,rayjobs -A`.

## Verification

`make test` with KubeRay checks that the `ray-job` example runs as `joe` with all 8 Ray tasks after Kueue admitted its cluster, that the dashboard and the API server answer at the dashboard's name from the host, that the dashboard points the browser at the API server there, that the finished job's driver output is in the dashboard's History, read through the dashboard as its pages do, that the finished job's Logs link leads to its cluster's Jobs page in Ray's dashboard for finished clusters, listing the job, and that a Ray cluster created through the API server shows its Ray dashboard through it, with its metrics in Prometheus; and that Ray's default dashboard is in Grafana's Ray folder for anonymous viewers. With Run:ai it checks that a Ray cluster in the project `test` is scheduled by `runai-scheduler` as one pod group, listed by Run:ai as a running `RayCluster` workload with its GPUs, and scraped by Prometheus ([Testing](testing.md)). The checks that run Ray (the example, the API server's cluster and their history, the Run:ai cluster) need the shared venv (`make frameworks`) and are skipped without it.


## Limitations

- No login: the dashboard and the API server serve anyone who reaches the controller, and the API server acts with its own service account, which can manage Ray resources in every namespace. Users' Kubernetes rights (their own namespace) apply to `kubectl`, not to the dashboard.
- The dashboard is KubeRay's early web UI: a finished job's **Logs** link reaches its cluster's Jobs list in Ray's dashboard, not the job's page.
- Ray's dashboard for finished clusters takes about 630 MiB on the controller and an 870 MiB image.
- History is kept only for Ray clusters with the collector; Ray clusters made with the dashboard's job form have none. Every namespace's Secret `ray-history-s3` holds the same keys, so users can read each other's Ray history in the bucket (as anyone can in the dashboard). Nothing expires: delete old sessions from the bucket by hand. A finished job's **Logs** link for a cluster without the collector (such as the dashboard's job form makes) lands on the History Server's not-found page.
- The KubeRay dashboard's Grafana links (Jobs, Clusters) come only from KubeRay's old v1 API; with the API server V2 it has none. Ray's own dashboard, live, has the Grafana panels.
- With KubeRay, Grafana is readable without a login (anonymous viewers), as embedding Ray's panels needs.
- With Run:ai, Ray clusters in project namespaces are created by the cluster admin (`bin/kubectl`), not by directory users such as `joe`, who have no Kubernetes rights there.
- With Run:ai, the History Server's Secret has another name in project namespaces (`genericsecret-ray-history-s3`, Run:ai's prefix), so a Ray manifest for a project differs from one for a user namespace in that name and the queue label.

## References

- [KubeRay](https://github.com/kuberay-project/kuberay), [documentation](https://docs.ray.io/en/latest/cluster/kubernetes/index.html), [Helm charts](https://github.com/ray-project/kuberay-helm), [API server](https://github.com/kuberay-project/kuberay/tree/master/apiserver), [dashboard](https://github.com/kuberay-project/kuberay/tree/master/dashboard)
- [Kueue: RayJob](https://kueue.sigs.k8s.io/docs/tasks/run/rayjobs/), [RayCluster](https://kueue.sigs.k8s.io/docs/tasks/run/rayclusters/)
- [HAProxy Kubernetes ingress](https://github.com/haproxytech/kubernetes-ingress)
