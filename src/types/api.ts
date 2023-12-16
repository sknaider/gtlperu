export interface loginBody {
  email: string;
  password: string;
}

export interface loginResponse {
  token: string;
}

export interface registerBody {
  email: string;
  password: string;
  is_active: boolean;
  is_superuser: boolean;
  is_verified: boolean;
  fullname: string;
}
