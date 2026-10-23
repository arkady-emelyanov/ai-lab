[← README](../README.md) · **KubeRay**

# KubeRay

## Overview

In k3s mode the lab runs [KubeRay](https://github.com/kuberay-project/kuberay), the Kubernetes operator for Ray, with its API server and its web dashboard (`kuberay: true`, the default; `kuberay: false` in `local.yml` leaves it out). Ray clusters, jobs and services are Kubernetes resources (`RayCluster`, `RayJob`, `RayService`); KubeRay starts their head and worker pods on the trays, Kueue admits them against the GPU quota, and the dashboard lists them in the browser.

KubeRay is installed from its release (1.7.1, `kuberay_version`): the operator and API server charts from `kuberay-project.github.io/kuberay-helm` and the images from `quay.io/kuberay`.

| Component | Where | Role |
|---|---|---|
| KubeRay operator (chart `kuberay-operator`) | namespace `kuberay`, on `sched-control` | reconciles `RayCluster`, `RayJob`, `RayService`, `RayCronJob`: head and worker pods, their services, job submission |
| KubeRay API server V2 (chart `kuberay-apiserver`) | namespace `kuberay`, on `sched-control` | Kubernetes' own API for Ray's resources and their events, and the Ray dashboards of KubeRay's services, for the dashboard |
| KubeRay dashboard (`quay.io/kuberay/dashboard`) | namespace `kuberay`, on `sched-control` | web UI: Ray jobs and clusters per namespace, their state, creating and deleting them, links to each cluster's Ray dashboard |
| `MutatingAdmissionPolicy` `rayjob-shutdown-after-finish` | cluster-wide | sets `shutdownAfterJobFinishes` on RayJobs created without it (the dashboard's job form), which Kueue requires |
| Ingress `kuberay` | HAProxy ingress (k3s add-on), `sched-control` port 80 | `http://kuberay.<controller address>.nip.io`: the API server's paths (`/apis/ray.io/`, `/apis/events.k8s.io/`, `/api/v1/`) to the API server, everything else to the dashboard |

The three use about 80 MiB on the controller (the ingress controller about 190 MiB more); Ray's own pods run on the trays.

## How it works

### Installation

The `kuberay` role (`roles/kuberay`, run by `make configure` after the k3s cluster configuration) writes one k3s manifest (`lab-kuberay.yaml` in k3s' manifests directory): the two charts as `HelmChart` resources, like the lab's other add-ons, the dashboard, which has no chart upstream, as a Deployment and Service, the ingress and the admission policy. All three services run on the control-plane node. Kueue starts managing Ray's resources by itself once their APIs exist.

### Dashboard and API server

The dashboard is a web application that the browser runs: it reads where the API server is from the dashboard's `/api/config` (`API_DOMAIN`), then calls the API server directly. Both are behind the same name, so the browser talks to one origin and no CORS is needed. The API server V2 passes requests for `ray.io` resources to Kubernetes with its own service account, and proxies `/api/v1/namespaces/<ns>/services/<service>/proxy/` for services KubeRay created, which is how the dashboard links a cluster's Ray dashboard (port 8265 of its head service).

Everything is plain HTTP: the lab has no certificates to import for it. HTTPS stays only where a component requires it (Run:ai, [NVIDIA Run:ai](runai.md)).

The dashboard opens on namespace `joe` (the first directory user); its namespace selector switches to others.

### Scheduling

Kueue manages `RayJob` and `RayCluster` (its Ray integrations are on by default): a Ray cluster waits suspended until Kueue admits all of its pods at once against the ClusterQueue `gpu`, like a JobSet ([Kubernetes](kubernetes.md)). The examples ask for queue `gpu`; a Ray resource without a queue label in a user namespace goes to its `default` queue. Kueue requires `shutdownAfterJobFinishes: true` on a RayJob, so its cluster's resources return to the queue when the job ends; a `MutatingAdmissionPolicy` (`rayjob-shutdown-after-finish`) sets it on RayJobs created without it, such as the dashboard's. Pod templates can carry Kueue's topology annotations (`kueue.x-k8s.io/podset-required-topology: accelerator.topograph.run/domain`) to keep a Ray cluster in one NVLink domain.

### Ray in the pods

KubeRay generates each Ray container's `ray start` command (`--block`, the head's dashboard on all addresses, the GPUs from the container's `nvidia.com/gpu` limit) and runs it with `bash -c`. The lab's examples use the image the other Kubernetes examples use (`buildpack-deps:noble`) with the shared venv on `/shared` first in `PATH`, so Ray, PyTorch and the fake GPU libraries are the same as for every other job (`make frameworks`). Images with Ray inside (`rayproject/ray`) work too.

## Usage

From the login node (`bin/ssh login`) as a directory user:

```
cd examples/kubernetes
./submit --wait ray-job.yaml           # RayJob: prints its name, writes the driver's output to <name>.out
kubectl get rayjobs,rayclusters,workloads
```

`ray-job.yaml` is the Ray example (`ray_demo.py`, one GPU task per GPU) as a `RayJob`: a head on one tray and a worker on the other, 4 GPUs each. KubeRay starts the cluster when Kueue admits it, submits the driver from a submitter pod (Job `<name>`, whose log is the driver's output), and deletes the cluster a minute after the job ends.

Open `http://kuberay.10.107.111.10.nip.io` (your bridge's prefix may differ): **Jobs** and **Clusters** list Ray's resources in the selected namespace, with their state; a running cluster links its Ray dashboard (tasks, actors, logs). Users can also create and delete Ray jobs and clusters there; the dashboard's job form fills in `rayproject/ray` as the image (several GB to pull the first time) and no queue label, so Kueue puts the job in the namespace's `default` queue.

As cluster admin from your machine: `bin/kubectl get rayclusters,rayjobs -A`.

## Verification

`make test` with KubeRay checks that the `ray-job` example runs as `joe` with all 8 Ray tasks after Kueue admitted its cluster, that the dashboard and the API server answer at the dashboard's name from the host, that the dashboard points the browser at the API server there, and that a Ray cluster created through the API server shows its Ray dashboard through it ([Testing](testing.md)).

## Limitations

- No login: the dashboard and the API server serve anyone who reaches the controller, and the API server acts with its own service account, which can manage Ray resources in every namespace. Users' Kubernetes rights (their own namespace) apply to `kubectl`, not to the dashboard.
- The dashboard is KubeRay's early web UI: its **History** pages need KubeRay's history server, which the lab does not run, and its jobs and clusters have no Grafana or log links.
- Ray's metrics are not in the lab's Prometheus or Grafana.
- KubeRay schedules through Kueue only: with Run:ai, Ray workloads do not go through KAI (KubeRay's `batchScheduler: kai-scheduler`).

## References

- [KubeRay](https://github.com/kuberay-project/kuberay), [documentation](https://docs.ray.io/en/latest/cluster/kubernetes/index.html), [Helm charts](https://github.com/ray-project/kuberay-helm), [API server](https://github.com/kuberay-project/kuberay/tree/master/apiserver), [dashboard](https://github.com/kuberay-project/kuberay/tree/master/dashboard)
- [Kueue: RayJob](https://kueue.sigs.k8s.io/docs/tasks/run/rayjobs/), [RayCluster](https://kueue.sigs.k8s.io/docs/tasks/run/rayclusters/)
- [HAProxy Kubernetes ingress](https://github.com/haproxytech/kubernetes-ingress)
