# One GPU VM on Nebius for the week-1 spike (E1-E4).
# Usage:
#   terraform plan            # preview, costs nothing
#   terraform apply           # create the VM (starts billing)
#   terraform apply -var stopped=true   # stop the GPU when you're done for the day
#   terraform destroy         # delete everything at the end

provider "nebius" {
  profile = {
    name = var.nebius_profile
  }
}

variable "nebius_profile" {
  description = "Nebius CLI profile to authenticate with"
  type        = string
  default     = "shivam"
}

variable "project_id" {
  description = "Nebius project to create resources in"
  type        = string
}

variable "subnet_id" {
  description = "Existing VPC subnet for the VM"
  type        = string
}

variable "platform" {
  description = "GPU platform (gpu-l40s-d = NVIDIA L40S on AMD hosts; gpu-l40s-a had no capacity on 2026-11-07)"
  type        = string
  default     = "gpu-l40s-d"
}

variable "preset" {
  description = "Size of the VM; smallest 1-GPU preset on gpu-l40s-d"
  type        = string
  default     = "1gpu-16vcpu-96gb"
}

variable "image_family" {
  description = "Boot image; ships an NVIDIA driver and CUDA 13"
  type        = string
  default     = "ubuntu24.04-cuda13-latest"
}

variable "ssh_user" {
  type    = string
  default = "dynagpu"
}

variable "ssh_public_key_path" {
  type    = string
  default = "~/.ssh/id_ed25519.pub"
}

variable "stopped" {
  description = "Set true to stop the VM (and its GPU billing) without deleting it"
  type        = bool
  default     = false
}

resource "nebius_compute_v1_disk" "boot" {
  parent_id           = var.project_id
  name                = "dynagpu-spike-boot"
  type                = "NETWORK_SSD"
  size_gibibytes      = 100
  block_size_bytes    = 4096
  source_image_family = { image_family = var.image_family }
}

resource "nebius_compute_v1_instance" "spike" {
  parent_id = var.project_id
  name      = "dynagpu-spike"
  stopped   = var.stopped

  resources = {
    platform = var.platform
    preset   = var.preset
  }

  boot_disk = {
    attach_mode   = "READ_WRITE"
    existing_disk = { id = nebius_compute_v1_disk.boot.id }
  }

  network_interfaces = [{
    name              = "eth0"
    subnet_id         = var.subnet_id
    ip_address        = {}
    public_ip_address = {}
  }]

  cloud_init_user_data = <<-EOT
    #cloud-config
    users:
      - name: ${var.ssh_user}
        sudo: ALL=(ALL) NOPASSWD:ALL
        shell: /bin/bash
        ssh_authorized_keys:
          - ${trimspace(file(pathexpand(var.ssh_public_key_path)))}
  EOT
}

output "public_ip" {
  value = try(nebius_compute_v1_instance.spike.status.network_interfaces[0].public_ip_address.address, null)
}

output "ssh" {
  value = try("ssh ${var.ssh_user}@${split("/", nebius_compute_v1_instance.spike.status.network_interfaces[0].public_ip_address.address)[0]}", null)
}
